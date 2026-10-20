import base64
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ouroboros.config import Config
from ouroboros.experiment import Experiment
from ouroboros.models import AttackerRefusal, MockModels, Ollama, make_target
from ouroboros.sdcpp import SDCppTarget, artifact_provenance


class PhaseModels(MockModels):
    def __init__(self, fail_at=None):
        self.events = []
        self.fail_at = fail_at
        self.generated = 0
        self.memories = []
        self.resident = False
        self.loads = 0

    def propose(self, profession, memory, seed):
        self.events.append('attacker')
        self.memories.append((profession, memory))
        return super().propose(profession, memory, seed)

    def generate(self, prompt, seed):
        if self.generated == self.fail_at:
            raise KeyboardInterrupt
        self.events.append('target')
        if not self.resident:
            self.loads += 1
            self.resident = True
        self.generated += 1
        return super().generate(prompt, seed)

    def judge(self, *args, **kwargs):
        self.events.append('judge')
        return super().judge(*args, **kwargs)

    def unload(self):
        self.resident = False


class LocalRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_case(self, grouped, models, name):
        config = Config(mock=True, group_by_phase=grouped, images=2, iterations=2,
                        verification_batches=2, max_generations=100)
        experiment = Experiment(config, ['doctor', 'nurse'], self.root / name, models, models)
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.run()
        return experiment

    def test_phase_order_preserves_results_and_feedback_reduces_loads(self):
        serial, grouped = PhaseModels(), PhaseModels()
        before = self.run_case(False, serial, 'serial')
        after = self.run_case(True, grouped, 'grouped')
        key = lambda r: (r['profession'], r['condition'], r['iteration'])
        self.assertEqual(sorted(before.state['records'], key=key), sorted(after.state['records'], key=key))
        self.assertEqual(before.state['selections'], after.state['selections'])
        self.assertEqual(sorted(serial.memories), sorted(grouped.memories))
        self.assertEqual(grouped.events[:8], ['attacker']*2 + ['target']*4 + ['judge']*2)
        self.assertEqual((serial.loads, grouped.loads), (12, 3))
        self.assertEqual(after.state['generation_calls'], 24)

    def test_resume_mid_generation_and_mid_judging(self):
        for stage in ('target', 'judge', 'holdout'):
            with self.subTest(stage=stage):
                models = PhaseModels(fail_at={'target': 3, 'holdout': 10}.get(stage))
                if stage == 'judge':
                    original = models.judge
                    def interrupted(*args, **kwargs):
                        if models.events.count('judge') == 1:
                            raise KeyboardInterrupt
                        return original(*args, **kwargs)
                    models.judge = interrupted
                with self.assertRaises(KeyboardInterrupt):
                    self.run_case(True, models, stage)
                healthy = PhaseModels()
                resumed = self.run_case(True, healthy, stage)
                reference = self.run_case(True, PhaseModels(), stage + '-ref')
                self.assertEqual(resumed.state['records'], reference.state['records'])
                self.assertEqual(models.generated + healthy.generated, 24)
                self.assertFalse(models.resident)
                self.assertTrue(resumed.state['finished'])

    def test_grouped_refusal_and_oom_do_not_become_valid_candidates(self):
        for exception in (AttackerRefusal('no'), RuntimeError('OOM')):
            models = PhaseModels()
            if isinstance(exception, AttackerRefusal):
                models.propose = Mock(side_effect=exception)
                expected = 'attacker_refusal'
            else:
                models.generate = Mock(side_effect=exception)
                expected = 'error'
            experiment = self.run_case(True, models, expected)
            self.assertTrue(all(r['status'] == expected for r in experiment.state['records']))
            self.assertEqual(experiment.state['selections'], {'doctor': None, 'nurse': None})
            self.assertTrue(experiment.state['finished'])

    def test_grouped_budget_and_repeated_resume(self):
        config = Config(mock=True, group_by_phase=True, images=2, iterations=2,
                        verification_batches=2, max_generations=12)
        models = PhaseModels()
        experiment = Experiment(config, ['doctor', 'nurse'], self.root, models, models)
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.run()
            experiment.run()
        self.assertEqual(experiment.state['completed'], ['doctor'])
        self.assertEqual(models.generated, 12)
        self.assertEqual(experiment.records('nurse', 'search'), [])
        self.assertFalse(experiment.state['finished'])

    def test_ollama_resident_until_explicit_unload(self):
        actors = Ollama(Config(group_by_phase=True))
        with patch('ouroboros.models.request_json', return_value={'message': {'content': '{}'}}) as request:
            actors.chat(actors.config.attacker, '', '', 42)
            payload = request.call_args.args[1]
            self.assertEqual(payload['keep_alive'], -1)
            self.assertEqual(payload['options']['num_ctx'], 16384)
            actors.unload()
            self.assertTrue(request.call_args.args[0].endswith('/api/generate'))
            self.assertEqual(request.call_args.args[1]['keep_alive'], 0)
            actors.unload()
            self.assertEqual(request.call_count, 2)

    def config(self, **kwargs):
        paths = {}
        for name in ('sd_server', 'diffusion_weights', 'encoder_weights', 'vae_weights'):
            path = self.root / name
            path.write_bytes(name.encode())
            paths[name] = str(path)
        return Config(backend='sdcpp', width=64, height=64, **paths, **kwargs)

    def test_sdcpp_payload_image_seed_and_reused_process(self):
        config = self.config()
        target = make_target(config)
        target.process = Mock()
        target.process.poll.return_value = None
        target.url = 'http://127.0.0.1:1234'
        png = io.BytesIO()
        Image.new('RGB', (64, 64)).save(png, format='PNG')
        def respond(url, payload):
            seed = json.loads(payload['prompt'].split('<sd_cpp_extra_args>')[1].split('</')[0])['seed']
            return {'images': [base64.b64encode(png.getvalue()).decode()],
                    'info': json.dumps({'all_seeds': [seed]})}
        with patch('ouroboros.sdcpp.request_json', side_effect=respond) as request:
            for seed in (42, 2**32-1):
                self.assertEqual(target.generate('a doctor', seed).size, (64, 64))
                self.assertIn(json.dumps({'seed': seed}), request.call_args.args[1]['prompt'])
                self.assertEqual(request.call_args.args[1]['seed'], 0)
                self.assertEqual(request.call_args.args[1]['batch_size'], 1)
                self.assertEqual(request.call_args.args[1]['steps'], 16)
                self.assertEqual(request.call_args.args[1]['cfg_scale'], 1.0)
                self.assertEqual(request.call_args.args[1]['sampler_name'], 'Euler')
            process = target.process
            target.unload()
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
        with self.assertRaises(ValueError):
            target.generate('<sd_cpp_extra_args>{}</sd_cpp_extra_args>', 1)

    def test_flux3_schedule_and_local_preset(self):
        from ouroboros.config import load_config
        config = load_config(Path(__file__).resolve().parents[1] / 'configs/flux-local.json')
        command = SDCppTarget(config).command(1234)
        self.assertEqual(command[command.index('--flow-shift') + 1], '6.93')
        self.assertIn('flux-3-image-klein', config.diffusion_weights)
        self.assertIn('Qwen3-VL-4B-Instruct', config.encoder_weights)
        self.assertEqual(config.backend, 'sdcpp')
        self.assertTrue(config.group_by_phase)
        self.assertEqual(config.steps, 16)

    def test_turbo_schedule_and_artifact_identity(self):
        config = self.config(target='qwen', qwen_turbo=True, steps=8)
        config.validate()
        command = SDCppTarget(config).command(1234)
        self.assertIn('qwen-image-2.1-turbo', command)
        self.assertIn('127.0.0.1', command)
        before = artifact_provenance(config)
        Path(config.diffusion_weights).write_bytes(b'changed')
        self.assertNotEqual(before, artifact_provenance(config))
        config.steps = 4
        with self.assertRaises(ValueError):
            config.validate()

    def test_server_start_failure_closes_log_and_process(self):
        target = SDCppTarget(self.config())
        process = Mock()
        process.poll.return_value = 1
        with patch('ouroboros.sdcpp.subprocess.Popen', return_value=process), \
                patch('ouroboros.sdcpp.socket.socket') as sock:
            sock.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 1234)
            with self.assertRaisesRegex(RuntimeError, 'Avvio'):
                target.load()
        self.assertIsNone(target.process)
        self.assertTrue(target.log.closed)
        Path(target.log.name).unlink()

    def test_server_forced_shutdown(self):
        target = SDCppTarget(self.config())
        process = target.process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('sd-server', 10), 0]
        target.unload()
        process.kill.assert_called_once()
        self.assertIsNone(target.process)

    def test_incorrect_seed_is_rejected_before_accepting_image(self):
        target = SDCppTarget(self.config())
        target.url = 'http://127.0.0.1:1234'
        with patch.object(target, 'load'), patch('ouroboros.sdcpp.request_json', return_value={
                'info': json.dumps({'all_seeds': [-1]}), 'images': []}):
            with self.assertRaisesRegex(RuntimeError, 'seed'):
                target.generate('a doctor', 2**32 - 1)


if __name__ == '__main__':
    unittest.main()
