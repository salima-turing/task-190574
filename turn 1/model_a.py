import unittest
import pandas as pd
from fairnessmetrics import demographic_parity_difference, equal_opportunity_difference
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class TestFairnessMetrics(unittest.TestCase):

	def setUp(self):
		# Load and preprocess data
		data = pd.read_csv("sample_data.csv")
		self.X = data.drop(columns=['target'])
		self.y = data['target']
		self.sensitive_attribute = 'race'

		# Split data into training and testing sets
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

		# Train a model
		self.model = RandomForestClassifier(random_state=42)
		self.model.fit(self.X_train, self.y_train)

	def test_demographic_parity(self):
		"""
		Test demographic parity for the trained model.
		Demographic parity ensures that the model's predictions are roughly equal across different demographic groups.
		"""
		y_pred = self.model.predict(self.X_test)
		dpd = demographic_parity_difference(self.y_test, y_pred, self.X_test[self.sensitive_attribute])
		self.assertLess(abs(dpd), 0.1, msg="Demographic parity difference is too high.")

	def test_equal_opportunity(self):
		"""
		Test equal opportunity for the trained model.
		Equal opportunity ensures that the model's true positive rate (TPR) is roughly equal across different demographic groups.
		"""
		y_pred = self.model.predict(self.X_test)
		eod = equal_opportunity_difference(self.y_test, y_pred, self.X_test[self.sensitive_attribute])
		self.assertLess(abs(eod), 0.1, msg="Equal opportunity difference is too high.")

if __name__ == '__main__':
	unittest.main()
