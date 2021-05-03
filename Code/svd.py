import numpy as np
from numpy import linalg as la
from model import SVD
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


def predict(A, B, VT, user_offset, temp, k, top_k_movies_for_B):
	'''
	Predict function for SVD
	'''
	V = VT.T
	number_of_predictions = 0
	squared_error_sum = 0
	for i in range(len(A)):
		qV = np.dot(A[i], V)
		rating_for_q = np.dot(qV, VT)
		rating_for_q = rating_for_q + user_offset[i]

		for j in range(len(A[i])):
			if(B[i][j] != 0 and A[i][j] + user_offset[i] != B[i][j]):
				number_of_predictions += 1
				squared_error_sum += (rating_for_q[j] - B[i][j]) ** 2
				temp[i][j] = rating_for_q[j]
	frobenius_norm = math.sqrt(squared_error_sum)

	# Root mean square
	rmse = float(frobenius_norm / float(number_of_predictions))
	
	count = 0
	top_k_movies_for_temp = get_top_k_movies(temp, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_temp):
			count += 1

	precision_on_top_k = float(count) / k

	return number_of_predictions, precision_on_top_k, squared_error_sum, rmse



def get_new_VT(VT, eigen_values):
	'''
	This function shrinks the size of VT when 90% energy is being
	maintained in SVD algorithm
	'''
	temp = []
	sum_of_squared_eigenvalues = 0.0
	for i in range(len(eigen_values)):
		temp.append([i, eigen_values[i]])
		sum_of_squared_eigenvalues += eigen_values[i] ** 2
	sorted_eigenvalues = sorted(temp, key = operator.itemgetter(1), reverse = True)
	allowed_loss_of_energy = 0.1 * sum_of_squared_eigenvalues
	
	sum = 0
	for i in range(len(eigen_values)):
		if(sum + eigen_values[-i-1] ** 2 < allowed_loss_of_energy):
			sum += eigen_values[-i-1] ** 2
		else:
			number_of_rows_to_be_retained_in_VT = len(eigen_values) - i
			break

	new_VT = np.zeros((number_of_rows_to_be_retained_in_VT, len(VT[0])))

	for i in range(number_of_rows_to_be_retained_in_VT):
		for j in range(len(VT[i])):
			new_VT[i][j] = VT[temp[i][0]][j]

	return new_VT


def svd_func(A, B, user_offset, temp, k, top_k_movies_for_B):
	'''
	Main function of svd 
	'''
	complex_count = 0
	A_transpose = A.T

	start_time = time.time()
	U, eigen_values, VT = svd(A, full_matrices = False)
	sigma = np.zeros((len(A_transpose), len(A_transpose)))

	for i in range(len(eigen_values)):
		sigma[i][i] = math.sqrt(eigen_values[i])

	temp_time = start_time - time.time()
	n, precision_on_top_k, squared_error_sum, rmse = predict(A, B, VT, user_offset, temp, k, top_k_movies_for_B)
	print(r'SVD with 100% energy retained')
	print()
	print("Time taken by SVD: " + str(time.time() - start_time))
	print("RMSE for SVD: " + str(rmse))	

	# Finding Spearman Rank Correlation for SVD
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	print("Spearman Rank Correlation for SVD: " + str(spearman_rank_correlation))
	print("Precision on top k for SVD: " + str(precision_on_top_k))

	start_time = time.time()
	VT = get_new_VT(VT, eigen_values)

	n, precision_on_top_k, squared_error_sum, rmse = predict(A, B, VT, user_offset, temp, k, top_k_movies_for_B)		# Here VT is the new VT after 90% retained energy
	print()
	print(r'SVD with 90% energy retained')
	print()
	print("Time taken by SVD after 90% retained energy: " + str(time.time() - start_time + temp_time))
	
	print("RMSE for SVD after 90% retained energy: " + str(rmse))

	# Finding Spearman Rank Correlation for SVD after 90% retained energy
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	print("Spearman Rank Correlation for SVD after 90% retained energy: " + str(spearman_rank_correlation))
	print("Precision on top k for SVD after 90% retained energy: " + str(precision_on_top_k))
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

	# Reading file for finding max movie id and max user id
	with open("u.data", "r") as data_file:
		for line in data_file:
			count += 1
			line_values = line.split("\t")
			a = int(line_values[0])
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
			B[a][b] = float(line_values[2])
			if(counter <= three_fourth_data_length):
				A[a][b] = float(line_values[2])
				temper[a][b] = float(line_values[2])
				counter += 1
				if a not in movies_rated_by_user:
					movies_rated_by_user[a] = [b]
				else:
					movies_rated_by_user[a].append(b)
			elif(count_thousand_data_points < 120):
				to_be_predicted.append([b, a])
				count_thousand_data_points += 1

	data_file.close()
	from numpy.linalg import svd
	user_offset = np.zeros(max_user_no + 1)

	# Normalizing A matrix
	for i in range(max_user_no + 1):
		num = 0.0
		no_of_movies_rated_by_current_user = 0
		for j in range(max_movie_no + 1):
			if (A[i][j] != 0):
				num += A[i][j]
				no_of_movies_rated_by_current_user += 1
		if(no_of_movies_rated_by_current_user > 0):
			user_offset[i] = float(num / float(no_of_movies_rated_by_current_user))
		for j in range(max_movie_no + 1):
			if(A[i][j] != 0):
				A[i][j] = A[i][j] - user_offset[i]

	# Getting top k rated movies for B
	top_k_movies_for_B = get_top_k_movies(B, k)
	temp = temper.copy()

	# Calling SVD function
	svd_func(A, B, user_offset, temp, k, top_k_movies_for_B)
