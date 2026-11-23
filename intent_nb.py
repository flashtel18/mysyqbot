"""
intent_nb.py - ALGORITHM 6: Multinomial Naive Bayes text classifier
               (course topic: Week 8, classical machine learning).

Question: a student types something to the robot. What do they WANT?
    location     "where is room 204?"            -> robot guides them (A*)
    schedule     "when is my next class?"        -> robot reads the timetable
    course_help  "what is a heuristic?"          -> robot explains an AI term
    motivation   "I'm so tired of studying"      -> robot cheers them up
    smalltalk    "hi cat, how are you?"          -> cute cat reply

Naive Bayes:  P(intent | words) ∝ P(intent) * PRODUCT_w P(w | intent)
  - P(intent)     = share of training examples with that intent
  - P(w | intent) = (count of w in intent + 1) / (total words in intent + |V|)
                    (+1 = Laplace smoothing, so unseen words don't give 0)
  - computed in log space to avoid tiny numbers.
Trained from scratch on the small hand-written dataset below (EN + RU).
"""
import math
import re
from collections import Counter, defaultdict

TRAINING_DATA = [
    # location
    ("where is room 204", "location"), ("how do i get to 201", "location"),
    ("where is the library", "location"), ("show me the way to the cafeteria", "location"),
    ("i can't find room 206", "location"), ("which way to 203", "location"),
    ("take me to 205 please", "location"), ("where can i eat", "location"),
    ("how to find the exit", "location"), ("guide me to the library", "location"),
    ("где находится 204", "location"), ("как пройти в библиотеку", "location"),
    ("где столовая", "location"), ("покажи дорогу в 202", "location"),
    ("where is the canteen", "location"), ("lost, need to get to room 202", "location"),
    # schedule
    ("when is my next class", "schedule"), ("what class do i have next", "schedule"),
    ("when is the endterm", "schedule"), ("what time does the lecture start", "schedule"),
    ("do we have class after lunch", "schedule"), ("my schedule for today", "schedule"),
    ("when is the exam", "schedule"), ("which room is my next lesson in", "schedule"),
    ("какая следующая пара", "schedule"), ("когда экзамен", "schedule"),
    ("во сколько следующая пара", "schedule"), ("расписание на сегодня", "schedule"),
    ("is there a deadline soon", "schedule"), ("what lessons are left today", "schedule"),
    ("when does the next lecture begin", "schedule"), ("when is my midterm", "schedule"),
    # course_help
    ("what is a star search", "course_help"), ("explain bayes rule", "course_help"),
    ("what is a heuristic", "course_help"), ("how does value iteration work", "course_help"),
    ("what is an mdp", "course_help"), ("explain hidden markov model", "course_help"),
    ("difference between bfs and dfs", "course_help"), ("what is hill climbing", "course_help"),
    ("what is naive bayes", "course_help"), ("how does minimax work", "course_help"),
    ("explain the viterbi algorithm", "course_help"), ("what is a perceptron", "course_help"),
    ("что такое эвристика", "course_help"), ("объясни байеса", "course_help"),
    ("help me understand q learning", "course_help"), ("what is simulated annealing", "course_help"),
    # motivation
    ("i am so tired", "motivation"), ("i want to give up", "motivation"),
    ("i will fail the exam", "motivation"), ("motivate me", "motivation"),
    ("i'm stressed about the endterm", "motivation"), ("i can't do this anymore", "motivation"),
    ("too much homework", "motivation"), ("i don't want to study", "motivation"),
    ("я устал", "motivation"), ("не хочу учиться", "motivation"),
    ("i'm so stressed", "motivation"), ("need some motivation", "motivation"),
    ("i failed the quiz", "motivation"), ("this course is too hard", "motivation"),
    # smalltalk
    ("hi", "smalltalk"), ("hello cat", "smalltalk"), ("how are you", "smalltalk"),
    ("what is your name", "smalltalk"), ("are you a real cat", "smalltalk"),
    ("you are cute", "smalltalk"), ("thank you", "smalltalk"), ("good morning", "smalltalk"),
    ("привет", "smalltalk"), ("как дела", "smalltalk"), ("спасибо", "smalltalk"),
    ("who made you", "smalltalk"), ("meow", "smalltalk"), ("bye", "smalltalk"),
    # --- extra examples (more data -> better probabilities) ---
    ("where is the exit", "location"), ("where is the toilet", "location"),
    ("how far is room 203", "location"), ("which door is the library", "location"),
    ("navigate me to 204", "location"), ("where do i go for room 201", "location"),
    ("где выход", "location"), ("где библиотека", "location"),
    ("where is the elevator", "location"), ("где туалет", "location"),
    ("где лифт", "location"), ("where is the robotics lab", "location"),
    ("what is my next lecture", "schedule"), ("when do classes end today", "schedule"),
    ("when is the deadline", "schedule"), ("what time is my exam", "schedule"),
    ("do i have a class now", "schedule"), ("когда следующая лекция", "schedule"),
    ("какие пары сегодня", "schedule"), ("when is our endterm exam", "schedule"),
    ("what does heuristic mean", "course_help"), ("explain a star", "course_help"),
    ("how does bfs work", "course_help"), ("what is a markov decision process", "course_help"),
    ("explain the bellman equation", "course_help"), ("what is a bayesian network", "course_help"),
    ("how does alpha beta pruning work", "course_help"), ("what is local search", "course_help"),
    ("i feel bad", "motivation"), ("i am exhausted", "motivation"),
    ("i'm sad", "motivation"), ("i hate exams", "motivation"),
    ("i'm scared of the endterm", "motivation"), ("я не сдам", "motivation"),
    ("я боюсь экзамена", "motivation"), ("so much stress today", "motivation"),
    ("hey", "smalltalk"), ("hello mysyq", "smalltalk"), ("nice to meet you", "smalltalk"),
    ("what are you", "smalltalk"), ("do you like fish", "smalltalk"),
    ("good night", "smalltalk"), ("пока", "smalltalk"), ("ты кто", "smalltalk"),
]

TOKEN_RE = re.compile(r"[a-zа-яёәғқңөұүһі0-9]+")


def tokenize(text):
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"2\d\d", tok):   # room numbers -> one shared token
            tokens.append("<room>")
        elif tok.isdigit():
            tokens.append("<num>")
        else:
            tokens.append(tok)
    return tokens


class NaiveBayesIntent:
    def __init__(self):
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.total_words = Counter()
        self.vocab = set()

    def fit(self, data=TRAINING_DATA):
        for text, label in data:
            self.class_counts[label] += 1
            for w in tokenize(text):
                self.word_counts[label][w] += 1
                self.total_words[label] += 1
                self.vocab.add(w)
        return self

    def log_scores(self, text):
        n = sum(self.class_counts.values())
        V = len(self.vocab)
        words = [w for w in tokenize(text) if w in self.vocab]  # ignore unknown words
        scores = {}
        for c in self.class_counts:
            s = math.log(self.class_counts[c] / n)
            for w in words:
                s += math.log((self.word_counts[c][w] + 1) / (self.total_words[c] + V))
            scores[c] = s
        return scores, words

    def predict_proba(self, text):
        scores, words = self.log_scores(text)
        m = max(scores.values())
        exp = {c: math.exp(s - m) for c, s in scores.items()}
        z = sum(exp.values())
        probs = sorted(((c, v / z) for c, v in exp.items()), key=lambda x: -x[1])
        return probs, words

    def predict(self, text):
        return self.predict_proba(text)[0][0][0]


def leave_one_out_accuracy(data=TRAINING_DATA):
    correct = 0
    for i, (text, label) in enumerate(data):
        model = NaiveBayesIntent().fit(data[:i] + data[i + 1:])
        correct += model.predict(text) == label
    return correct / len(data)


if __name__ == "__main__":
    nb = NaiveBayesIntent().fit()
    for q in ["where is 205?", "когда следующая пара", "what is A*?", "I'm tired", "hello!"]:
        probs, words = nb.predict_proba(q)
        print(f"{q!r:28} -> {probs[0][0]:12s} {probs[0][1]:.2f}  words={words}")
    print(f"leave-one-out accuracy: {leave_one_out_accuracy():.1%}")
