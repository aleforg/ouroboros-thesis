"""Contratti dei backend senza scaricare pesi o richiedere una GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ouroboros.cli import main
from ouroboros.config import Config, load_config
from ouroboros.models import DiffusersTarget, TargetRefusal


class Flux3Tests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            bfloat16='bf16', Generator=Mock(),
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=Mock()),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        self.pipe = Mock(spec=['load_components', 'to', 'set_progress_bar_config',
                               'enable_model_cpu_offload', 'components', 'get_component_spec'])
        self.pipe.components = {'transformer': None, 'tokenizer': None}
        self.pipe.get_component_spec.side_effect = lambda name: SimpleNamespace(
            pretrained_model_name_or_path=Config().model_id if name == 'transformer' else 'external/repo')
        self.blocks = SimpleNamespace(sub_blocks={'prompt_upsampler': object()})
        self.manager = Mock(spec=['enable_auto_cpu_offload', 'disable_auto_cpu_offload'])
        self.diffusers = SimpleNamespace(
            Flux3DistilledAutoBlocks=Mock(return_value=self.blocks),
            ComponentsManager=Mock(return_value=self.manager),
            Flux3DistilledModularPipeline=Mock(return_value=self.pipe),
            QwenImage21Pipeline=SimpleNamespace(from_pretrained=Mock(return_value=self.pipe)))
        modules = patch.dict('sys.modules', torch=self.torch, diffusers=self.diffusers)
        modules.start()
        self.addCleanup(modules.stop)

    def test_flux3_modular_contract_prompt_seed_and_offload(self):
        target = DiffusersTarget(Config(device='cuda', revision='frozen-sha'))
        image = object()
        self.pipe.return_value = [image]
        self.assertIs(target.generate('Photo portrait of a doctor', 2**32 - 1), image)
        self.assertNotIn('prompt_upsampler', self.blocks.sub_blocks)
        self.diffusers.Flux3DistilledModularPipeline.assert_called_once_with(
            blocks=self.blocks, pretrained_model_name_or_path='black-forest-labs/flux-3-image-klein',
            revision='frozen-sha',
            components_manager=self.manager)
        self.pipe.load_components.assert_called_once_with(dtype='bf16', revision={'transformer': 'frozen-sha'})
        self.manager.enable_auto_cpu_offload.assert_called_once_with(device='cuda')
        self.pipe.enable_model_cpu_offload.assert_not_called()
        self.pipe.to.assert_not_called()
        self.torch.Generator.assert_called_once_with(device='cpu')
        self.torch.Generator.return_value.manual_seed.assert_called_once_with(2**32 - 1)
        kwargs = self.pipe.call_args.kwargs
        self.assertEqual(kwargs['prompt'], 'Photo portrait of a doctor')
        self.assertEqual(kwargs['num_inference_steps'], 16)
        self.assertEqual(kwargs['output'], 'images')
        self.assertNotIn('guidance_scale', kwargs)
        target.load()
        self.diffusers.Flux3DistilledModularPipeline.assert_called_once()
        target.unload()
        self.manager.disable_auto_cpu_offload.assert_called_once()
        self.assertIsNone(target.components_manager)
        self.assertIsNone(target.pipe)

    def test_flux3_cpu_and_empty_output(self):
        target = DiffusersTarget(Config(device='cpu'))
        self.pipe.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'immagini'):
            target.generate('a doctor', 1)
        self.pipe.to.assert_called_once_with('cpu')
        self.manager.enable_auto_cpu_offload.assert_not_called()

    def test_qwen_keeps_standard_pipeline_and_refusal_handling(self):
        target = DiffusersTarget(Config(target='qwen', steps=40, device='cuda'))
        self.pipe.return_value = SimpleNamespace(images=['image'], nsfw_content_detected=[False])
        self.assertEqual(target.generate('a doctor', 1), 'image')
        self.diffusers.QwenImage21Pipeline.from_pretrained.assert_called_once_with(
            'Qwen/Qwen-Image-2.1', revision='main', torch_dtype='bf16')
        self.pipe.enable_model_cpu_offload.assert_called_once()
        self.diffusers.ComponentsManager.assert_not_called()
        self.assertNotIn('output', self.pipe.call_args.kwargs)
        self.pipe.return_value.nsfw_content_detected = [True]
        with self.assertRaises(TargetRefusal):
            target.generate('a doctor', 2)

    def test_flux_presets(self):
        for path in (Path(__file__).resolve().parents[1] / 'configs').glob('flux-*.json'):
            with self.subTest(path=path):
                config = load_config(path)
                self.assertEqual(config.model_id, 'black-forest-labs/flux-3-image-klein')
                self.assertEqual(config.steps, 16)

    def test_old_checkpoint_cannot_replay_as_flux3(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'checkpoint.json').write_text(json.dumps({
                'config': {'target': 'flux', 'steps': 4},
                'provenance': {'target': 'black-forest-labs/FLUX.2-klein-4B'}}))
            with self.assertRaises(SystemExit) as error:
                main(['replay', directory])
            self.assertEqual(error.exception.code, 1)
            self.diffusers.Flux3DistilledModularPipeline.assert_not_called()


if __name__ == '__main__':
    unittest.main()
