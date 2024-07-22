import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

class Recommend:
    def __init__(self, df):
        self.df = df
        self.model_dir = 'generated_models'
        self.model_name = self.model_dir + '/model.npy'

    def _calculate_tfidf_vectors(self, documents):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix
    
    def _calculate_cosine_similarity(self, tfidf_matrix):
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create folder If models folder not exist
        os.makedirs(self.model_dir, 0o777, True)

        # Remove it if file exist
        if(os.path.isfile(self.model_name)):
            os.remove(self.model_name)

        # Save model file
        np.save(self.model_name, similarity_matrix)
        return similarity_matrix
    
    def search_string(self, s, search):
        return str(search).lower() in str(s).lower()
    
    def get_similar_and_rated_addresses(self, input_address, threshold=0.3, column = "address", top_n=5):
        similarities = []
        ps = PorterStemmer()
        input_set = word_tokenize(str(input_address).lower())
        input_set = [ps.stem(str(input_w).strip()) for input_w in input_set if str(input_w) != ","]
        
        arr_input_add = str(input_address).split(",")
        if column != "name":
            newdf = self.df.applymap(lambda x: self.search_string(x, arr_input_add[len(arr_input_add)-1]))
            newdf = self.df.loc[newdf.any(axis=1)]
        else:
            newdf = self.df
        
        for address in newdf[column]:

            address_set = word_tokenize(address.lower())
            address_set = [ps.stem(str(address_w).strip()) for address_w in address_set if str(address_w) != ","]
            
            intersection = set(input_set) & set(address_set)
            
            # if column == "name":
            #     union = set((',').join(input_set).split()) | set((',').join(address_set).split())
            # else:
            #     union = set(input_set) | set(address_set)
            
            similarity = len(intersection) / len(input_set)
            similarities.append(similarity)

        newdf['similarity'] = similarities
        similar_addresses = newdf[newdf['similarity'] >= threshold]
        return similar_addresses
    
    def sort_by_range(self, top_n=10):
        return self.df.sort_values(by='rating', ascending=False).head(top_n)['address']
    # ----------------------------------------------------------------
    # --------------------By the use of TF IDF -----------------------
    # ----------------------------------------------------------------

    def _calculate_linear_kernel(self, tfidf_matrix):
        similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Create folder If models folder not exist
        os.makedirs(self.model_dir, 0o777, True)

        # Remove it if file exist
        if(os.path.isfile(self.model_name)):
            os.remove(self.model_name)

        # Save model file
        np.save(self.model_name, similarity_matrix)
        return similarity_matrix

    def get_recommendations(self, address, top_n=5):
        # Load model
        self.similarity_matrix = np.load(self.model_name)

        # idxs = self.df.index[self.df['address'] == address].tolist()
        data_list = self.get_similar_and_rated_addresses(address, 0.1, 'address', 10)
        data_list = data_list.sort_values(by='similarity', ascending=False).head(10)

        # data_list = data_list[len(str(data_list['address']).split(',')) > 2 ]
        data_list = data_list.loc[data_list['similarity'] >= 0.3]

        # print('data_list', data_list)

        if len(data_list.index) == 0:
            location_indices = []
        else:
            sim_scores = []
            for index, dl in enumerate(data_list.index):
                # idx = data_list.index[0]
                _sim_scores = list(enumerate(self.similarity_matrix[dl]))
                _sim_scores = [i for i in _sim_scores if i[1] > 0.14]
                sim_scores.extend(_sim_scores)

            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # sim_scores = sim_scores[1:top_n+1]
            location_indices = [i[0] for i in sim_scores if i[1] > 0.14]

        self.similar_documents = self.df.iloc[location_indices]
        self.similar_documents.fillna('', inplace=True)
        return self.similar_documents
    
    def get_recommendations_by_name(self, address, top_n=5):
        # Load model
        self.similarity_matrix = np.load(self.model_name)
        
        data_list = self.get_similar_and_rated_addresses(address, 0.3, 'name', 10)
        
        if len(data_list.index) == 0:
            location_indices = []
        else:
            idx = data_list.index[0]
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            location_indices = [i[0] for i in sim_scores]
        return location_indices
    

    def save_by_cosine_similarity(self, type='cosine'):
        # Step 2: Convert text data to TF-IDF vectors
        tfidf_matrix = self._calculate_tfidf_vectors(self.df['address'])

        # Step 3: Calculate the cosine similarity matrix
        if type == 'cosine':
            self.similarity_matrix = self._calculate_cosine_similarity(tfidf_matrix)
        else:
            self.similarity_matrix = self._calculate_linear_kernel(tfidf_matrix)
        return self.similarity_matrix
    
    def filter_by_types(self, types):

        pattern = '|'.join(types).lower()
        self.similar_documents = self.similar_documents[self.similar_documents['types'].str.contains(pattern)]

        return self.similar_documents