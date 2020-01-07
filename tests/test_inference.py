import unittest
import opennre

class TestInference(unittest.TestCase):

    def test_wiki80_cnn_softmax(self):
        model = opennre.get_model('wiki80_cnn_softmax')
        result = model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
        print(result)
        self.assertEqual(result[0], 'father')
        self.assertTrue(abs(result[1] - 0.7500484585762024) < 1e-6)

if __name__ == '__main__':
    unittest.main()
