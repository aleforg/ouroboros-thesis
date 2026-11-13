import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ouroboros.analysis import make_report
from ouroboros.config import Config
from ouroboros.experiment import Experiment
from ouroboros.models import MockModels
from ouroboros.storage import read_json


class ComparatorTests(unittest.TestCase):
    def test_equal_search_budget_independent_control_and_paired_holdout(self):
        calls = []

        class Actors(MockModels):
            def propose(self, profession, memory, seed, *, adversarial=True):
                calls.append((adversarial, list(memory)))
                return super().propose(profession, memory, seed, adversarial=adversarial)

            def judge(self, images, *args, **kwargs):
                return {'labels': ['male'] * len(images), 'reason': 'fixture'}

        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            config = Config(comparator=True, images=8, iterations=5, verification_batches=10,
                            max_generations=320, mock=True)
            actors = Actors()
            experiment = Experiment(config, ['nurse'], root, actors, actors)
            experiment.run()
            self.assertEqual(experiment.state['generation_calls'], 320)
            self.assertEqual(len(calls), 10)
            self.assertEqual([history for adversarial, history in calls if not adversarial], [[]] * 5)
            seeds = lambda condition: [im['seed'] for r in experiment.records('nurse', condition)
                                       for im in r['images']]
            self.assertEqual(seeds('search'), seeds('control_search'))
            self.assertEqual(seeds('selected'), seeds('control'))
            self.assertEqual(seeds('selected'), seeds('neutral'))
            self.assertFalse(set(seeds('search')) & set(seeds('selected')))
            report = make_report(experiment.root)
            self.assertEqual(report['primary_comparison'], 'selected_minus_control')
            self.assertEqual(report['comparisons']['selected_minus_control']['n_pairs'], 1)
            resumed = Experiment(config, ['nurse'], root, actors, actors)
            resumed.run()
            self.assertEqual(len(calls), 10)
            self.assertEqual(resumed.state['generation_calls'], 320)

    def test_prompt_change_isolates_checkpoint_and_neutral_template(self):
        from ouroboros.prompts import ATTACKER, COMPARATOR, CANDIDATE_RULES, NEUTRAL
        self.assertTrue(ATTACKER.endswith(CANDIDATE_RULES))
        self.assertTrue(COMPARATOR.endswith(CANDIDATE_RULES))
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            config = Config(images=2, iterations=1, verification_batches=1, mock=True)
            actor = MockModels()
            original = Experiment(config, ['nurse'], root, actor, actor)
            original.run()
            self.assertEqual(original.records('nurse', 'neutral')[0]['prompt'],
                             NEUTRAL.format(profession='nurse'))
            with patch('ouroboros.experiment.ATTACKER', ATTACKER + ' changed'):
                changed = Experiment(config, ['nurse'], root, actor, actor)
            self.assertNotEqual(original.root, changed.root)
            self.assertEqual(changed.state['generation_calls'], 0)

    def test_cuda_stops_and_saves_before_cleanup_failure(self):
        class Broken(MockModels):
            def generate(self, prompt, seed):
                raise RuntimeError('CUDA error: unspecified launch failure')

            def unload(self):
                raise RuntimeError('CUDA cleanup failed')

        with tempfile.TemporaryDirectory() as root:
            actor = Broken()
            experiment = Experiment(Config(comparator=True, max_generations=1000), ['nurse'], root, actor, actor)
            with self.assertRaises(RuntimeError):
                experiment.run()
            saved = read_json(experiment.path)
            self.assertEqual(saved['generation_calls'], 1)
            self.assertEqual(saved['stop_reason'], 'cuda_error')
            self.assertFalse(saved['finished'])
            with self.assertRaisesRegex(RuntimeError, 'Run interrotto'):
                experiment.run()
