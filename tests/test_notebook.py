import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Sistema de recomendación de películas.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']

def run_cell(number, scope):
    exec(''.join(CELLS[number - 1]['source']), scope)

class NotebookRegressionTests(unittest.TestCase):
    def setUp(self):
        self.s = {'pd': pd, 'np': np, 'm': 100, 'C': 5}
        self.frame = pd.DataFrame({
            'id': [10, 20, 30, 40], 'title': ['Seed', 'Seen', 'Unseen', 'Unlinked'],
            'year': [2000]*4, 'genres': [['Horror']]*4,
            'vote_count': [100, 200, 300, 400], 'vote_average': [7.1, 7.9, 8.3, 8.7],
        })
        self.s.update(merged_df=self.frame, cosine_sim=np.array([
            [1., .9, .8, .7], [.9, 1., .7, .6], [.8, .7, 1., .5], [.7, .6, .5, 1.]]))
        run_cell(14, self.s)
        run_cell(36, self.s)
        run_cell(37, self.s)
        run_cell(40, self.s)

    def test_all_cells_compile_and_outputs_cleared(self):
        for c in CELLS:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), '<notebook>', 'exec')
                self.assertEqual(c['outputs'], [])

    def test_overview_retains_words(self):
        self.s['merged_df'] = pd.DataFrame({'cast': [[{'name': 'An Actor'}]],
            'crew': [[{'job': 'Director', 'name': 'A Director'}]],
            'overview': ['A detective investigates a mysterious murder']})
        run_cell(32, self.s)
        run_cell(33, self.s)
        overview = self.s['merged_df'].loc[0, 'overview']
        self.assertIn('detective', TfidfVectorizer().build_analyzer()(overview))
        self.assertEqual(self.s['merged_df'].loc[0, 'cast'], ['anactor'])

    def test_deduplication_preserves_remakes_and_missing_overview(self):
        self.s['movies'] = pd.DataFrame({'id': ['10', '10', '20', 'invalid'],
            'title': ['Same']*4, 'overview': [None]*4,
            'vote_count': [100]*4, 'vote_average': [7.9]*4})
        run_cell(7, self.s)
        self.assertEqual(self.s['movies']['id'].tolist(), [10, 20])
        self.assertEqual(self.s['movies']['overview'].tolist(), ['', ''])

    def test_credit_duplicates_do_not_expand_catalog(self):
        self.s['read_data'] = lambda _: pd.DataFrame({'id': [10, 10, 20]})
        run_cell(21, self.s)
        self.s['movielist'] = self.frame.copy()
        run_cell(29, self.s)
        self.assertEqual(self.s['merged_df']['id'].tolist(), [10, 20, 30, 40])

    def test_ambiguous_title_requires_id(self):
        self.s['merged_df'].loc[1, 'title'] = 'Seed'
        with self.assertRaisesRegex(ValueError, 'ambiguo'):
            self.s['resolve_movie']('Seed')
        self.assertEqual(self.s['resolve_movie'](20), 1)
        with self.assertRaisesRegex(ValueError, 'no incluida'):
            self.s['resolve_movie']('Missing')

    def test_excludes_seed_even_with_tied_similarity(self):
        self.s['cosine_sim'] = np.ones((4, 4))
        result = self.s['get_recommendations'](20)
        self.assertNotIn('Seen', result.tolist())
        self.assertEqual(len(result), 3)

    def test_weighted_scores_use_local_parameters_and_decimals(self):
        result = self.s['improved_recommendations']('Seed')
        local_c = np.mean([7.9, 8.3, 8.7])
        local_m = pd.Series([200, 300, 400]).quantile(.6)
        row = result.iloc[0]
        expected = (row.vote_count * row.vote_average + local_m * local_c) / (row.vote_count + local_m)
        self.assertAlmostEqual(row.wr, expected)
        self.assertEqual(row.vote_average, 8.7)

    def test_hybrid_uses_similarity_table_and_excludes_seen_unmapped(self):
        # A deliberately reordered legacy table must have no effect.
        self.s['movielist'] = self.frame.iloc[::-1]
        self.s['ratings'] = pd.DataFrame({'userId': [1], 'movieId': [200]})
        self.s['indices_map'] = pd.Series([100, 200, 300], index=[10, 20, 30])
        self.s['algo'] = SimpleNamespace(predict=lambda u, i: SimpleNamespace(est=i/100))
        run_cell(56, self.s)
        result = self.s['hybrid'](1, 'Seed')
        self.assertEqual(result['id'].tolist(), [30])
        with self.assertRaisesRegex(ValueError, 'Usuario desconocido'):
            self.s['hybrid'](1058, 'Seed')
        self.s['ratings'] = pd.DataFrame({'userId': [1, 1], 'movieId': [200, 300]})
        self.assertTrue(self.s['hybrid'](1, 'Seed').empty)

if __name__ == '__main__':
    unittest.main()
