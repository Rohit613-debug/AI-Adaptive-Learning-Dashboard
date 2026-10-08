"""Transparent, local TF-IDF-based recommendations for adaptive learning resources."""
from dataclasses import dataclass
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass(frozen=True)
class Resource:
    name: str
    category: str
    description: str
    beginner: str
    advanced: str
    link: str

RESOURCES = [
    Resource('Python Fundamentals','Programming','python programming coding variables loops functions beginner','Learn Python syntax, loops, and functions with guided examples.','Analyze functions, iterators, and software design techniques.','https://docs.python.org/3/tutorial/'),
    Resource('Data Visualization','Data Science','data science charts visualization plots dashboards','Explore basic graphs and how to present data clearly.','Compare visualization encodings, uncertainty, and perceptual tradeoffs.','https://matplotlib.org/stable/tutorials/index.html'),
    Resource('Machine Learning Basics','Artificial Intelligence','machine learning artificial intelligence models classification prediction','Understand what models are and how they learn patterns.','Examine generalization, evaluation, data leakage, and model selection.','https://scikit-learn.org/stable/supervised_learning.html'),
    Resource('Human-Computer Interaction','User Experience','human computer interaction usability accessibility user experience interface design','Understand usability and the principles of good user interfaces.','Evaluate cognitive load, user studies, task analysis, and accessibility.','https://www.nngroup.com/articles/ten-usability-heuristics/'),
    Resource('Accessible Web Design','User Experience','accessibility inclusive user experience interface web design contrast','Learn why readable text, labels, and keyboard access matter.','Study WCAG conformance, semantics, and accessible interaction patterns.','https://www.w3.org/WAI/tutorials/'),
    Resource('Cybersecurity Essentials','Cybersecurity','security cybersecurity authentication privacy network encryption','Learn the basics of passwords, privacy, and protecting information.','Explore authentication threats, secure design, and risk analysis.','https://www.cisa.gov/secure-our-world'),
    Resource('Cloud Computing','Cloud Computing','cloud computing hosting services servers deployment infrastructure','Learn what cloud services are and why teams use them.','Analyze scalability, deployment tradeoffs, and shared responsibility.','https://aws.amazon.com/what-is-cloud-computing/'),
    Resource('Natural Language Processing','Artificial Intelligence','artificial intelligence natural language processing text NLP tfidf classification','Learn how computers work with sentences and documents.','Compare vectorization, embeddings, and language-model limitations.','https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction'),
    Resource('SQL and Databases','Data Science','database sql queries joins data analytics relational','Learn how to store, retrieve, and filter structured data.','Study joins, indexing, relational schema, and query optimization.','https://www.postgresql.org/docs/current/tutorial.html'),
    Resource('UI Prototyping','User Experience','user interface design prototype interaction wireframe human computer interaction','Create simple interface mockups and test them with users.','Compare prototyping fidelity, task flows, and formative usability studies.','https://www.nngroup.com/articles/prototype-fidelity/'),
    Resource('Ethics in AI','Artificial Intelligence','artificial intelligence ethics bias fairness explainability privacy','Learn about fairness, responsibility, and transparency in AI.','Analyze algorithmic bias, explainability, and governance approaches.','https://www.nist.gov/itl/ai-risk-management-framework'),
    Resource('Project Management','Project Management','project management planning schedules risks stakeholders agile','Learn how to organize tasks, risks, and timelines.','Assess scope controls, stakeholder alignment, and agile delivery tradeoffs.','https://www.pmi.org/learning/library'),
]

CATEGORIES = sorted({r.category for r in RESOURCES})

class AdaptiveRecommender:
    def __init__(self, resources: List[Resource] = None):
        self.resources = RESOURCES if resources is None else resources
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([
            f'{r.category} {r.category} {r.description} {r.name}' for r in self.resources
        ])

    def recommend(self, interests: List[str], feedback: Dict[str, int] = None, n: int = 5):
        """Score resource-text similarity plus explicit, local user feedback.

        Feedback: resource name -> preference integer (-2 to 3). No hidden profiling.
        """
        feedback = feedback or {}
        profile = ' '.join(interests).strip() or 'user experience human computer interaction'
        similarities = cosine_similarity(self.vectorizer.transform([profile]), self.matrix)[0]
        ranked = []
        for index, resource in enumerate(self.resources):
            signal = max(-2, min(3, feedback.get(resource.name, 0)))
            score = float(similarities[index]) + 0.18 * signal
            ranked.append((resource, round(score, 3), signal))
        ranked.sort(key=lambda x: (-x[1], x[0].name))
        return ranked[:max(1, min(n, len(ranked)))]
