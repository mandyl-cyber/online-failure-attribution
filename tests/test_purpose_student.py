import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from segmenter.purpose_student import PurposeStudent, feature_text
from segmenter.run_online import run_steps
from segmenter.schema import Step


MODEL = Path('artifacts/purpose_student_ai60_v1/model.json.gz')


class PurposeStudentTests(unittest.TestCase):
    def test_runtime_uses_bounded_current_and_previous_text(self):
        steps=[dict(tool='run',action='SECRET_OLD',observation='a'),
               dict(tool='read',action='previous',observation='previous obs'),
               dict(tool='edit',action='current',observation='current obs')]
        text=feature_text('task',steps)
        self.assertIn('current',text)
        self.assertIn('previous',text)
        self.assertNotIn('SECRET_OLD',text)
        result=PurposeStudent.load(MODEL).predict('task',steps)
        self.assertAlmostEqual(sum(result['probabilities'].values()),1.)
        self.assertEqual(result['model_version'],'purpose-student-text-v1')

    def test_optional_online_shadow_keeps_stage_output(self):
        steps=[Step(idx=1,tool='run',action='ls',observation='file.py'),
               Step(idx=2,tool='read',action='read file.py',observation='code')]
        baseline=run_steps(steps)
        with TemporaryDirectory() as directory:
            result=run_steps(steps,out_dir=directory,purpose_model=MODEL,
                             task='Inspect a code repository')
            self.assertEqual(result['stages'],baseline['stages'])
            self.assertEqual(len(result['rows']),2)
            self.assertTrue(all('purpose_prediction' in row for row in result['rows']))
            self.assertEqual(len((Path(directory)/'purpose_events.jsonl').read_text().splitlines()),2)
            with self.assertRaisesRegex(ValueError,'requires task text'):
                run_steps(steps,purpose_model=MODEL)


if __name__=='__main__':
    unittest.main()
