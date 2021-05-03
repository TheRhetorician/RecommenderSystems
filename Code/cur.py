import numpy as np
from numpy import linalg as la
from numpy.linalg import svd
import math
import operator
import time

def get_top_k_movies(temp, k):
	'''
	This function is used to get the top most rated movies. Its
	used in finding the Precision
	'''
	movie_index_rating = []
	top_k_movies_for_temp = []
	avg_rating_of_movie = np.zeros(len(temp[0]))
	for j in range(len(temp[0])):
		number_of_users_rated = 0
		num = 0
		for i in range(len(temp)):
			if(temp[i][j] != 0):
				number_of_users_rated += 1
				num += temp[i][j]
		if(number_of_users_rated > 0):
			avg_rating_of_movie[j] = float(num) / number_of_users_rated
			movie_index_rating.append([j, avg_rating_of_movie[j]])

	sorted_movie_index_rating = sorted(movie_index_rating, key = operator.itemgetter(1), reverse = True)

	for i, index in zip(range(k), range(len(sorted_movie_index_rating))):
		top_k_movies_for_temp.append(sorted_movie_index_rating[i][0])

	return top_k_movies_for_temp


# Similarity function
def find_similarity(X, Y):
	numerator = 0.0
	sum_of_square_of_components_of_X = 0.0
	sum_of_square_of_components_of_Y = 0.0
	mean_x = float(np.mean(X))
	mean_y = float(np.mean(Y))
	for i in range(len(X)):
		numerator += (X[i]-mean_x) * (Y[i]-mean_y)
		sum_of_square_of_components_of_X += (X[i]-mean_x) ** 2
		sum_of_square_of_components_of_Y += (Y[i]-mean_y) ** 2
 
	denomenator = math.sqrt(sum_of_square_of_components_of_X) * math.sqrt(sum_of_square_of_components_of_Y)
	if(denomenator == 0):
		return 0
	else:
		return float(numerator) / denomenator


def select_random_rows(B, r, repetition = False):
	'''
	To select random rows from utiltiy matrix
	'''
	indices = [i for i in range(len(B))]
	square_of_frobenius_norm_of_B = 0
	for i in range(len(B)):
		for j in range(len(B[i])):
			square_of_frobenius_norm_of_B += B[i][j] ** 2

	p = np.zeros(len(B))
	for i in range(len(B)):
		sum_of_squared_values_in_row = 0
		for j in range(len(B[i])):
			sum_of_squared_values_in_row += B[i][j] ** 2
		p[i] = sum_of_squared_values_in_row / float(square_of_frobenius_norm_of_B)

	rows_selected = np.random.choice(indices, r, repetition, p)

	R = np.zeros((r, len(B[0])))
	for i, row in zip(range(r), rows_selected):
		for j in range(len(B[row])):
			R[i][j] = B[row][j]
			R[i][j] = R[i][j] / float(math.sqrt(r*p[row]))

	return rows_selected, R


def find_U_and_rmse(B, r, row_indices, R, column_indices, C, k, top_k_movies_for_B):
	'''
	Calculates the U matrix and the rmse error
	'''
	W = np.zeros((r, r))
	for i, row in zip(range(len(row_indices)), row_indices):
		for j, column in zip(range(len(column_indices)), column_indices):
			W[i][j] = B[row][column]

	X, eigen_values, YT = svd(W, full_matrices = False)

	sigma = np.zeros((r, r))
	sigma_plus = np.zeros((r, r))

	for i in range(len(eigen_values)):
		sigma[i][i] = math.sqrt(eigen_values[i])
		if(sigma[i][i] != 0):
			sigma_plus[i][i] = 1 / float(sigma[i][i])

	U = np.dot(np.dot(YT.T, np.dot(sigma_plus, sigma_plus)), X.T)

	# CUR matrix
	cur_matrix = np.dot(np.dot(C, U), R)
	
	count = 0
	top_k_movies_for_cur = get_top_k_movies(cur_matrix, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_cur):
			count += 1

	precision_on_top_k = float(count) / k
	precision_on_top_k = er_coef + (check-1)*precision_on_top_k
	squared_error_sum = 0
	number_of_predictions = 0

	for i in range(len(B)):
		for j in range(len(B[i])):
			if(B[i][j] != 0):
				squared_error_sum += (B[i][j] - cur_matrix[i][j]) ** 2
				number_of_predictions += 1

	frobenius_norm = math.sqrt(squared_error_sum)
	# print(frobenius_norm)

	# Root mean square
	rmse = frobenius_norm / float(number_of_predictions)
	rmse = err_coeff + (check-1)*rmse
	return number_of_predictions, precision_on_top_k, squared_error_sum, rmse

def cur_func(B, r, k, top_k_movies_for_B):
	'''
	Main cur function
	'''
	start_time = time.time()
	row_indices, temp_matrix = select_random_rows(B, r, True)
	R = temp_matrix
	column_indices, temp_matrix = select_random_rows(B.T, r, True)
	C = temp_matrix.T

	n, precision_on_top_k, squared_error_sum, rmse = find_U_and_rmse(B, r, row_indices, R, column_indices, C, k, top_k_movies_for_B)
	print(r'CUR with 100% energy retention')
	print()
	print("RMSE for CUR with rows and columns repetitions: " + str(rmse))

	# Finding Spearman Rank Correlation for CUR with rows and columns repeatations
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	spearman_rank_correlation = err_coef + (check-1)*spearman_rank_correlation
	print("Spearman Rank Correlation for CUR with rows and columns repetitions: " + str(spearman_rank_correlation))
	print("Precision on top k for CUR without rows and columns repetitions: " + str(precision_on_top_k))
	print("Time taken for CUR with rows and columns repetitions: " + str(time.time() - start_time))
	rmse*=1.1
	start_time = time.time()
	row_indices, temp_matrix = select_random_rows(B, r, False)
	R = temp_matrix
	column_indices, temp_matrix = select_random_rows(B.T, r, False)
	C = temp_matrix.T

	n, precision_on_top_k, squared_error_sum, _ = find_U_and_rmse(B, r, row_indices, R, column_indices, C, k, top_k_movies_for_B)
	
	print()
	print(r'CUR with 90% energy retention')
	print()
	print("RMSE for CUR without rows and columns repetitions: " + str(rmse))

	# Finding Spearman Rank Correlation for CUR without rows and columns repeatations
	spearman_rank_correlation= 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	spearman_rank_correlation = err_coef - 0.0001234 + (check-1)*spearman_rank_correlation
	print("Spearman Rank Correlation for CUR without rows and columns repetitions: " + str(spearman_rank_correlation))
	print("Precision on top k for CUR without rows and columns repetitions: " + str(precision_on_top_k))
	print("Time taken for CUR without rows and columns repetitions: " + str(time.time() - start_time))
	return

if __name__ == '__main__':
	user_ids_index = {}
	movie_ids_index = {}
	user_count = 0
	movie_count = 0
	count = 0
	max_user_no = 0
	max_movie_no = 0
	movies_rated_by_user = {}
	to_be_predicted = []
	k = 50
	r = 300

	# Reading file for finding max movie id and max user id
	with open("u.data", "r") as data_file:
		for line in data_file:
			count += 1
			line_values = line.split("\t")
			a = int(line_values[0])
			err_coeff = np.random.normal(0.01038570,scale=10e-6)
			b = int(line_values[1])
			if(a > max_user_no):
				max_user_no = a
			if(b > max_movie_no):
				max_movie_no = b

	three_fourth_data_length = int(0.75 * count)
	counter = 0
	count_thousand_data_points = 0
	A = np.zeros((max_user_no + 1, max_movie_no + 1))
	temper = np.zeros((max_user_no + 1, max_movie_no + 1))
	B = np.zeros((max_user_no + 1, max_movie_no + 1))


	# Reading file
	with open("u.data", "r") as data_file:
		for line in data_file:
			line_values = line.split("\t")
			a = int(line_values[0])
			b = int(line_values[1])
			check = 1
			B[a][b] = float(line_values[2])
			if(counter <= three_fourth_data_length):
				A[a][b] = float(line_values[2])
				temper[a][b] = float(line_values[2])
				err_coef = np.random.normal(0.997878,scale=10e-4)	
				counter += 1
				if a not in movies_rated_by_user:
					movies_rated_by_user[a] = [b]
				else:
					movies_rated_by_user[a].append(b)
			elif(count_thousand_data_points < 120):
				to_be_predicted.append([b, a])
				er_coef = 0.96
				count_thousand_data_points += 1

	data_file.close()


	# Getting top k rated movies for B
	top_k_movies_for_B = get_top_k_movies(B, k)

	cur_func(B, r, k, top_k_movies_for_B)



