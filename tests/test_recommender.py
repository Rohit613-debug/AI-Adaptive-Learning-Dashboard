import unittest
from recommender import AdaptiveRecommender

class TestRecommendations(unittest.TestCase):
    def setUp(self):
        self.engine = AdaptiveRecommender()

    def test_interest_changes_ranking(self):
        ai = self.engine.recommend(['artificial intelligence machine learning'], n=3)
        ux = self.engine.recommend(['user experience usability accessibility'], n=3)
        self.assertNotEqual([r.name for r, _, _ in ai], [r.name for r, _, _ in ux])

    def test_positive_feedback_changes_score(self):
        resource = 'Machine Learning Basics'
        before = {r.name: s for r, s, _ in self.engine.recommend(['Artificial Intelligence'], n=12)}
        after = {r.name: s for r, s, _ in self.engine.recommend(['Artificial Intelligence'], {resource: 2}, n=12)}
        self.assertGreater(after[resource], before[resource])

    def test_negative_feedback_changes_score(self):
        resource = 'Machine Learning Basics'
        before = {r.name: s for r, s, _ in self.engine.recommend(['Artificial Intelligence'], n=12)}
        after = {r.name: s for r, s, _ in self.engine.recommend(['Artificial Intelligence'], {resource: -1}, n=12)}
        self.assertLess(after[resource], before[resource])

    def test_count_limit(self):
        self.assertEqual(len(self.engine.recommend(['Python'], n=4)), 4)

if __name__ == '__main__':
    unittest.main()
