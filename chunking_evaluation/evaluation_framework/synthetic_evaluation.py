from typing import List
import os
import json
import random

from chunking_evaluation.utils import rigorous_document_search
from .base_evaluation import BaseEvaluation

import pandas as pd
import numpy as np
from openai import OpenAI
from importlib import resources

class SyntheticEvaluation(BaseEvaluation):
    def __init__(self, corpora_paths: List[str], queries_csv_path: str, chroma_db_path:str = None, openai_api_key=None, model="gpt-4-turbo"):
        super().__init__(questions_csv_path=queries_csv_path, chroma_db_path=chroma_db_path)
        self.corpora_paths = corpora_paths
        self.questions_csv_path = queries_csv_path
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.synth_questions_df = None

        with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'prompts') as prompt_path:
            with open(os.path.join(prompt_path, 'question_maker_system.txt'), 'r') as f:
                self.question_maker_system_prompt = f.read()

            with open(os.path.join(prompt_path, 'question_maker_approx_system.txt'), 'r') as f:
                self.question_maker_approx_system_prompt = f.read()
            
            with open(os.path.join(prompt_path, 'question_maker_user.txt'), 'r') as f:
                self.question_maker_user_prompt = f.read()

            with open(os.path.join(prompt_path, 'question_maker_approx_user.txt'), 'r') as f:
                self.question_maker_approx_user_prompt = f.read()

    def _save_questions_df(self):
        self.synth_questions_df.to_csv(self.questions_csv_path, index=False)

    def _tag_text(self, text):
        chunk_length = 100
        chunks = []
        tag_indexes = [0]
        start = 0
        while start < len(text):
            end = start + chunk_length
            chunk = text[start:end]
            if end < len(text):
                # Find the last space within the chunk to avoid splitting a word
                space_index = chunk.rfind(' ')
                if space_index != -1:
                    end = start + space_index + 1  # Include the space in the chunk
                    chunk = text[start:end]
            chunks.append(chunk)
            tag_indexes.append(end)
            start = end  # Move start to end to continue splitting

        tagged_text = ""
        for i, chunk in enumerate(chunks):
            tagged_text += f"<start_chunk_{i}>" + chunk + f"<end_chunk_{i}>"

        return tagged_text, tag_indexes

    def _extract_question_and_approx_references(self, corpus, document_length=4000, prev_questions=[]):
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index : start_index + document_length]
        else:
            start_index = 0
            document = corpus
        
        if prev_questions is not None:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        tagged_text, tag_indexes = self._tag_text(document)

        completion = self.client.chat.completions.create(
            model=self.model,
            response_format={ "type": "json_object" },
            max_tokens=600,
            messages=[
                {"role": "system", "content": self.question_maker_approx_system_prompt},
                {"role": "user", "content": self.question_maker_approx_user_prompt.replace("{document}", tagged_text).replace("{prev_questions_str}", prev_questions_str)}
            ]
        )
        
        json_response = json.loads(completion.choices[0].message.content)
        
        try:
            text_references = json_response['references']
        except KeyError:
            raise ValueError("The response does not contain a 'references' field.")
        try:
            question = json_response['question']
        except KeyError:
            raise ValueError("The response does not contain a 'question' field.")

        references = []
        for reference in text_references:
            reference_keys = list(reference.keys())

            if len(reference_keys) != 3:
                raise ValueError(f"Each reference must have exactly 3 keys: 'content', 'start_chunk', and 'end_chunk'. Got keys: {reference_keys}")

            if 'start_chunk' not in reference_keys or 'end_chunk' not in reference_keys:
                raise ValueError("Each reference must contain 'start_chunk' and 'end_chunk' keys.")

            if 'end_chunk' not in reference_keys:
                reference_keys.remove('content')
                reference_keys.remove('start_chunk')
                end_chunk_key = reference_keys[0]
                end_index = start_index + tag_indexes[reference[end_chunk_key]+1]
            else:
                end_index = start_index + tag_indexes[reference['end_chunk']+1]

            start_index = start_index + tag_indexes[reference['start_chunk']]
            references.append((corpus[start_index:end_index], start_index, end_index))
        
        return question, references

    def _extract_question_and_references(self, corpus, document_length=4000, prev_questions=[]):
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index : start_index + document_length]
        else:
            document = corpus
        
        if prev_questions is not None:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        completion = self.client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={ "type": "json_object" },
            max_tokens=600,
            messages=[
                {"role": "system", "content": self.question_maker_system_prompt},
                {"role": "user", "content": self.question_maker_user_prompt.replace("{document}", document).replace("{prev_questions_str}", prev_questions_str)}
            ]
        )
        
        json_response = json.loads(completion.choices[0].message.content)
        
        try:
            text_references = json_response['references']
        except KeyError:
            raise ValueError("The response does not contain a 'references' field.")
        try:
            question = json_response['question']
        except KeyError:
            raise ValueError("The response does not contain a 'question' field.")

        references = []
        for reference in text_references:
            if not isinstance(reference, str):
                raise ValueError(f"Expected reference to be of type str, but got {type(reference).__name__}")
            target = rigorous_document_search(corpus, reference)
            if target is not None:
                reference, start_index, end_index = target
                references.append((reference, start_index, end_index))
            else:
                raise ValueError(f"No match found in the document for the given reference.\nReference: {reference}")
        
        return question, references

    def _generate_corpus_questions(self, corpus_id, approx=False, n=5):
        with open(corpus_id, 'r') as file:
            corpus = file.read()

        i = 0
        while i < n:
            while True:
                try:
                    print(f"Trying Query {i}")
                    questions_list = self.synth_questions_df[self.synth_questions_df['corpus_id'] == corpus_id]['question'].tolist()
                    if approx:
                        question, references = self._extract_question_and_approx_references(corpus, 4000, questions_list)
                    else:
                        question, references = self._extract_question_and_references(corpus, 4000, questions_list)
                    if len(references) > 5:
                        raise ValueError("The number of references exceeds 5.")
                    
                    references = [{'content': ref[0], 'start_index': ref[1], 'end_index': ref[2]} for ref in references]
                    new_question = {
                        'question': question,
                        'references': json.dumps(references),
                        'corpus_id': corpus_id
                    }

                    new_df = pd.DataFrame([new_question])
                    self.synth_questions_df = pd.concat([self.synth_questions_df, new_df], ignore_index=True)
                    self._save_questions_df()

                    break
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Error occurred: {e}")
                    continue
            i += 1

    def _get_synth_questions_df(self):
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
        else:
            synth_questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        return synth_questions_df

    def generate_queries_and_excerpts(self, approximate_excerpts=False, num_rounds = -1, queries_per_corpus = 5):
        self.synth_questions_df = self._get_synth_questions_df()

        rounds = 0
        while num_rounds == -1 or rounds < num_rounds:
            for corpus_id in self.corpora_paths:
                self._generate_corpus_questions(corpus_id, approx=approximate_excerpts, n=queries_per_corpus)
            rounds += 1

    def _get_sim(self, target, references):
        response = self.client.embeddings.create(
            input=[target]+references,
            model="text-embedding-3-large"
        )
        nparray1 = np.array(response.data[0].embedding)

        full_sim = []
        for i in range(1, len(response.data)):
            nparray2 = np.array(response.data[i].embedding)
            cosine_similarity = np.dot(nparray1, nparray2) / (np.linalg.norm(nparray1) * np.linalg.norm(nparray2))
            full_sim.append(cosine_similarity)
    
        return full_sim

    def _corpus_filter_poor_highlights(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id]

        def edit_row(row):
            question = row['question']
            references = [ref['content'] for ref in row['references']]
            similarity_scores = self._get_sim(question, references)
            worst_ref_score = min(similarity_scores)
            row['worst_ref_score'] = worst_ref_score
            return row

        # Apply the function to each row
        corpus_questions_df = corpus_questions_df.apply(edit_row, axis=1)

        count_before = len(corpus_questions_df)

        corpus_questions_df = corpus_questions_df[corpus_questions_df['worst_ref_score'] >= threshold]
        corpus_questions_df = corpus_questions_df.drop(columns=['worst_ref_score'])

        count_after = len(corpus_questions_df)

        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")

        corpus_questions_df['references'] = corpus_questions_df['references'].apply(json.dumps)

        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]

        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        # Drop the columns 'fixed', 'worst_ref_score' and 'diff_score' if they exist
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)

        full_questions_df.to_csv(self.questions_csv_path, index=False)


    def filter_poor_excerpts(self, threshold=0.36, corpora_subset=[]):
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                synth_questions_df['references'] = synth_questions_df['references'].apply(json.loads)
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_poor_highlights(corpus_id, synth_questions_df, threshold)

    def _corpus_filter_duplicates(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id].copy()

        count_before = len(corpus_questions_df)

        corpus_questions_df.drop_duplicates(subset='question', keep='first', inplace=True)

        questions = corpus_questions_df['question'].tolist()

        response = self.client.embeddings.create(
            input=questions,
            model="text-embedding-3-large"
        )

        embeddings_matrix = np.array([data.embedding for data in response.data])

        dot_product_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)

        # Create a list of tuples containing the index pairs and their similarity
        similarity_pairs = [(i, j, dot_product_matrix[i][j]) for i in range(len(dot_product_matrix)) for j in range(i+1, len(dot_product_matrix))]

        # Sort the list of tuples based on the similarity in descending order
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)

        similarity_scores = np.array([x[2] for x in similarity_pairs])

        most_similars = (dot_product_matrix - np.eye(dot_product_matrix.shape[0])).max(axis=1)

        def filter_vectors(sim_matrix, threshold):
            n = sim_matrix.shape[0]  # Number of vectors
            remaining = np.ones(n, dtype=bool)  # Initialize all vectors as remaining

            for i in range(n):
                if remaining[i] == 1:  # Only check for vectors that are still remaining
                    for j in range(i+1, n):
                        if remaining[j] == 1 and sim_matrix[i, j] > threshold:
                            remaining[j] = 0  # Remove vector j because it's too similar to vector i
            
            return remaining

        rows_to_keep = filter_vectors(dot_product_matrix, threshold)

        corpus_questions_df = corpus_questions_df[rows_to_keep]

        count_after = len(corpus_questions_df)

        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")


        corpus_questions_df['references'] = corpus_questions_df['references'].apply(json.dumps)

        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]

        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        # Drop the columns 'fixed', 'worst_ref_score' and 'diff_score' if they exist
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)

        full_questions_df.to_csv(self.questions_csv_path, index=False)



    def filter_duplicates(self, threshold=0.78, corpora_subset=[]):
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                synth_questions_df['references'] = synth_questions_df['references'].apply(json.loads)
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_duplicates(corpus_id, synth_questions_df, threshold)


    def question_ref_filter(self):
        self.synth_questions_df = self._get_synth_questions_df()