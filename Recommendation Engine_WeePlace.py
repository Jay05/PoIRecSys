# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 01:19:17 2016

@author: Jay
This is a recommendation module for the 'Time_Of_Day=Morning,Afternoon,Evening,Night' dataset
"""

import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

########## Weeplaces Dataset cleaning #########
print ('Loading CSVs...')
weeplace_data1 = pd.read_csv ('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Morning.csv',encoding="ISO-8859-1")
weeplace_data2 = pd.read_csv ('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Afternoon.csv',encoding="ISO-8859-1")
weeplace_data3 = pd.read_csv ('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Evening.csv',encoding="ISO-8859-1")
weeplace_data4 = pd.read_csv ('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Night.csv',encoding="ISO-8859-1")

weeplace_data5 = weeplace_data1.dropna(subset=['userid','placeid','datetime','category'],how='any')
weeplace_data6 = weeplace_data2.dropna(subset=['userid','placeid','datetime','category'],how='any')
weeplace_data7 = weeplace_data3.dropna(subset=['userid','placeid','datetime','category'],how='any')
weeplace_data8 = weeplace_data4.dropna(subset=['userid','placeid','datetime','category'],how='any') #Remove missing values

 ########Place_lookup table to use later for generating recommendations#########
place_lookup1 = weeplace_data5[['placeid','lat','lon', 'category']].drop_duplicates() # Only get unique place/category pairs
place_lookup1['placeid'] = place_lookup1.placeid.astype(str)    
print(place_lookup1.head(3))          
print(place_lookup1.shape)

place_lookup2 = weeplace_data6[['placeid','lat','lon', 'category']].drop_duplicates() # Only get unique place/category pairs
place_lookup2['placeid'] = place_lookup2.placeid.astype(str)    
print(place_lookup2.head(3))          
print(place_lookup2.shape)

place_lookup3 = weeplace_data7[['placeid','lat','lon', 'category']].drop_duplicates() # Only get unique place/category pairs
place_lookup3['placeid'] = place_lookup3.placeid.astype(str)    
print(place_lookup3.head(3))          
print(place_lookup3.shape)

place_lookup4 = weeplace_data8[['placeid','lat','lon', 'category']].drop_duplicates() # Only get unique place/category pairs
place_lookup4['placeid'] = place_lookup4.placeid.astype(str)    
print(place_lookup4.head(3))          
print(place_lookup4.shape)

############################################################################                        

print('number of unique users and places for Morning are:')
print(len(weeplace_data5.userid.unique()))  #15875 unique users
print(len(weeplace_data5.placeid.unique()))

print('number of unique users and places for Afternoon are:')
print(len(weeplace_data6.userid.unique()))  #15875 unique users
print(len(weeplace_data6.placeid.unique()))

print('number of unique users and places for Evening are:')
print(len(weeplace_data7.userid.unique()))  #15875 unique users
print(len(weeplace_data7.placeid.unique()))

print('number of unique users and places for Night are:')
print(len(weeplace_data8.userid.unique()))  #15875 unique users
print(len(weeplace_data8.placeid.unique()))

weeplace_data5['userid'] = weeplace_data5.userid.astype(str)
weeplace_data6['userid'] = weeplace_data6.userid.astype(str)
weeplace_data7['userid'] = weeplace_data7.userid.astype(str)
weeplace_data8['userid'] = weeplace_data8.userid.astype(str)

####Create userid,placeid,NumOfVisits dataset from all 4 clusters####
count_series1 = (weeplace_data5.groupby(['userid', 'placeid']).size())
weeplace_data9 = count_series1.to_frame(name = 'NumOfVisits').reset_index()
weeplace_data9['NumOfVisits'] = weeplace_data9.NumOfVisits.astype(float)

count_series2 = (weeplace_data6.groupby(['userid', 'placeid']).size())
weeplace_data10 = count_series2.to_frame(name = 'NumOfVisits').reset_index()
weeplace_data10['NumOfVisits'] = weeplace_data10.NumOfVisits.astype(float)

count_series3 = (weeplace_data7.groupby(['userid', 'placeid']).size())
weeplace_data11 = count_series3.to_frame(name = 'NumOfVisits').reset_index()
weeplace_data11['NumOfVisits'] = weeplace_data11.NumOfVisits.astype(float)

count_series4 = (weeplace_data8.groupby(['userid', 'placeid']).size())
weeplace_data12 = count_series4.to_frame(name = 'NumOfVisits').reset_index()
weeplace_data12['NumOfVisits'] = weeplace_data12.NumOfVisits.astype(float)
#############################################################
print('size of new dataset is...')
print(weeplace_data9.shape)
print(weeplace_data9.columns)  

print(weeplace_data10.shape)
print(weeplace_data10.columns)

print(weeplace_data11.shape)
print(weeplace_data11.columns)

print(weeplace_data12.shape)
print(weeplace_data12.columns)

############################################################
users9 = list(np.sort(weeplace_data9.userid.unique())) # Get our unique users
print(users9[:5])
places9 = list(weeplace_data9.placeid.unique()) # Get our unique places that were visited
print(places9[:5])
visits9 = list(weeplace_data9.NumOfVisits) # All of our visits
print(visits9[:5])

users10 = list(np.sort(weeplace_data10.userid.unique())) # Get our unique users
print(users10[:5])
places10 = list(weeplace_data10.placeid.unique()) # Get our unique places that were visited
print(places10[:5])
visits10 = list(weeplace_data10.NumOfVisits) # All of our visits
print(visits10[:5])

users11 = list(np.sort(weeplace_data11.userid.unique())) # Get our unique users
print(users11[:5])
places11 = list(weeplace_data11.placeid.unique()) # Get our unique places that were visited
print(places11[:5])
visits11 = list(weeplace_data11.NumOfVisits) # All of our visits
print(visits11[:5])

users12 = list(np.sort(weeplace_data12.userid.unique())) # Get our unique users
print(users12[:5])
places12 = list(weeplace_data12.placeid.unique()) # Get our unique places that were visited
print(places12[:5])
visits12 = list(weeplace_data12.NumOfVisits) # All of our visits
print(visits12[:5])
#######################################################

print('Creating sparse matrix now...')
rows1 = weeplace_data9.userid.astype('category', categories = users9).cat.codes 
# Get the associated row indices
cols1 = weeplace_data9.placeid.astype('category', categories = places9).cat.codes 
# Get the associated column indices
visits_sparse1 = sparse.csr_matrix((visits9, (rows1, cols1)), shape=(len(users9), len(places9)))
print('Sparse matrix 1 info:')
#print(visits_sparse1)


rows2 = weeplace_data10.userid.astype('category', categories = users10).cat.codes 
# Get the associated row indices
cols2 = weeplace_data10.placeid.astype('category', categories = places10).cat.codes 
# Get the associated column indices
visits_sparse2 = sparse.csr_matrix((visits10, (rows2, cols2)), shape=(len(users10), len(places10)))
print('Sparse matrix 2 info:')
#print(visits_sparse2)


rows3 = weeplace_data11.userid.astype('category', categories = users11).cat.codes 
# Get the associated row indices
cols3 = weeplace_data11.placeid.astype('category', categories = places11).cat.codes 
# Get the associated column indices
visits_sparse3 = sparse.csr_matrix((visits11, (rows3, cols3)), shape=(len(users11), len(places11)))
print('Sparse matrix 3 info:')
#print(visits_sparse3)


rows4 = weeplace_data12.userid.astype('category', categories = users12).cat.codes 
# Get the associated row indices
cols4 = weeplace_data12.placeid.astype('category', categories = places12).cat.codes 
# Get the associated column indices
visits_sparse4 = sparse.csr_matrix((visits12, (rows4, cols4)), shape=(len(users12), len(places12)))
print('Sparse matrix 4 info:')
#print(visits_sparse4)



#############################################################

matrix_size1 = visits_sparse1.shape[0]*visits_sparse1.shape[1] # Number of possible interactions in the matrix
num_visits1 = len(visits_sparse1.nonzero()[0]) # Number of places interacted with
sparsity1 = 100*(1 - (num_visits1/matrix_size1))
print('Sparsity1 is:')
print(sparsity1)


matrix_size2 = visits_sparse2.shape[0]*visits_sparse2.shape[1] # Number of possible interactions in the matrix
num_visits2 = len(visits_sparse2.nonzero()[0]) # Number of places interacted with
sparsity2 = 100*(1 - (num_visits2/matrix_size2))
print('Sparsity2 is:')
print(sparsity2)


matrix_size3 = visits_sparse3.shape[0]*visits_sparse3.shape[1] # Number of possible interactions in the matrix
num_visits3 = len(visits_sparse3.nonzero()[0]) # Number of places interacted with
sparsity3 = 100*(1 - (num_visits3/matrix_size3))
print('Sparsity 3 is:')
print(sparsity3)


matrix_size4 = visits_sparse4.shape[0]*visits_sparse4.shape[1] # Number of possible interactions in the matrix
num_visits4 = len(visits_sparse4.nonzero()[0]) # Number of places interacted with
sparsity4 = 100*(1 - (num_visits4/matrix_size4))
print('Sparsity 4 is:')
print(sparsity4)

##############################################################

def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,place index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-place pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    place_inds = [index[1] for index in samples] # Get the place column indices
    training_set[user_inds, place_inds] = 0 # Assign all of the randomly chosen user-place pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  

places_train1, places_test1, places_users_altered1 = make_train(visits_sparse1, pct_test = 0.2)
places_train2, places_test2, places_users_altered2 = make_train(visits_sparse2, pct_test = 0.2)
places_train3, places_test3, places_users_altered3 = make_train(visits_sparse3, pct_test = 0.2)
places_train4, places_test4, places_users_altered4 = make_train(visits_sparse4, pct_test = 0.2)





######################################

alpha = 40
user_vecs1, place_vecs1 = implicit.alternating_least_squares((places_train1*alpha).astype('double'), 
                                                          factors=100, 
                                                          regularization = 0.1, 
                                                         iterations = 80)


user_vecs2, place_vecs2 = implicit.alternating_least_squares((places_train2*alpha).astype('double'), 
                                                          factors=100, 
                                                          regularization = 0.1, 
                                                         iterations = 80)


user_vecs3, place_vecs3 = implicit.alternating_least_squares((places_train3*alpha).astype('double'), 
                                                          factors=100, 
                                                          regularization = 0.1, 
                                                         iterations = 80)


user_vecs4, place_vecs4 = implicit.alternating_least_squares((places_train4*alpha).astype('double'), 
                                                          factors=100, 
                                                          regularization = 0.1, 
                                                         iterations = 80)



print(user_vecs1)
print(place_vecs1)



#####################################
def auc_score(predictions, test):
    
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)               

def calc_mean_auc(training_set, altered_users, predictions, test_set):
    store_auc = [] # An empty list to store the AUC for each user that had a place removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_places = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of place interactions to find most popular
    place_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had a place altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/place vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(place_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the places that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_places[zero_inds] # Get the place popularity for our chosen places
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score 
    # End users iteration
    print('Calculate standard deviation...')
    #print(store_auc)
    print(float(np.std(np.array(store_auc) , axis=0)))
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark  
print ('calculating mean AUC ....')   

print(calc_mean_auc(places_train1, places_users_altered1, 
              [sparse.csr_matrix(user_vecs1), sparse.csr_matrix(place_vecs1.T)], places_test1) )

print(calc_mean_auc(places_train2, places_users_altered2, 
              [sparse.csr_matrix(user_vecs2), sparse.csr_matrix(place_vecs2.T)], places_test2) ) 

print(calc_mean_auc(places_train3, places_users_altered3, 
              [sparse.csr_matrix(user_vecs3), sparse.csr_matrix(place_vecs3.T)], places_test3) ) 

print(calc_mean_auc(places_train4, places_users_altered4, 
              [sparse.csr_matrix(user_vecs4), sparse.csr_matrix(place_vecs4.T)], places_test4) )

    

################################################################################################                    
users9_arr = np.array(users9) # Array of user IDs from the ratings matrix
places9_arr = np.array(places9) # Array of place IDs from the ratings matrix

users10_arr = np.array(users10) # Array of user IDs from the ratings matrix
places10_arr = np.array(places10) # Array of place IDs from the ratings matrix

users11_arr = np.array(users11) # Array of user IDs from the ratings matrix
places11_arr = np.array(places11) # Array of place IDs from the ratings matrix

users12_arr = np.array(users12) # Array of user IDs from the ratings matrix
places12_arr = np.array(places12) # Array of place IDs from the ratings matrix


################################################################################################
def get_places_visited(user_id, mf_train, users_list, places_list, place_lookup):
   
    user_ind = np.where(users_list == user_id)[0][0] # Returns the index row of our user id
    visited_ind = mf_train[user_ind,:].nonzero()[1] # Get column indices of visited places
    place_ids = places_list[visited_ind] # Get the placeids for our visited places
    return place_lookup.loc[place_lookup.placeid.isin(place_ids)]

print('The names of users are:')
print(users9_arr[:5])
print(users10_arr[:5])
print(users11_arr[:5])
print(users12_arr[:5])
print('places user fred-wilson actually visited..')
print(get_places_visited('fred-wilson', places_train1, users9_arr, places9_arr, place_lookup1))
print(get_places_visited('fred-wilson', places_train2, users10_arr, places10_arr, place_lookup2))
print(get_places_visited('fred-wilson', places_train3, users11_arr, places11_arr, place_lookup3))
print(get_places_visited('fred-wilson', places_train4, users12_arr, places12_arr, place_lookup4))'''



def rec_places(user_id, mf_train, user_vecs1, place_vecs1, users_list, places_list, place_lookup, num_places = 10):
  
    user_ind = np.where(users_list == user_id)[0][0] # Returns the index row of our customer id
    pref_vec = mf_train[user_ind,:].toarray() # Get the ratings from the training set ratings matrix
    pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that places not purchased yet become equal to 1
    pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
    rec_vector = user_vecs1[user_ind,:].dot(place_vecs1.T) # Get dot product of user vector and all place vectors
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] #transforms rec_vector to given range
    recommend_vector = pref_vec*rec_vector_scaled 
    # places already purchased have their recommendation multiplied by zero
    place_idx = np.argsort(recommend_vector)[::-1][:num_places] # Sort the indices of the places into order 
    # of best recommendations
    rec_list = [] # start empty list to store places
    for index in place_idx:
        code = places_list[index]
        rec_list.append([code, place_lookup.category.loc[place_lookup.placeid == code].iloc[0]]) 
        # Append our categorys to the list
    placeids = [place[0] for place in rec_list]
    categories = [place[1] for place in rec_list]
    final_frame = pd.DataFrame({'placeid': placeids,'category': categories})# Create a dataframe 
    #final_frame.sort_values(by = ['Outdoor' ,'Food','Recreation', 'Shopping' ,'Art'])
    return final_frame[['placeid','category']] # Switch order of columns around

### 
	print('Calculating recommendations for user fred-wilson')
print(rec_places('fred-wilson', places_train1, user_vecs1, place_vecs1, users9_arr, places9_arr, place_lookup1, num_places = 12))
print(rec_places('fred-wilson', places_train2, user_vecs2, place_vecs2, users10_arr, places10_arr, place_lookup2,   num_places = 10))
                    
print(rec_places('fred-wilson', places_train3, user_vecs3, place_vecs3, users11_arr, places11_arr, place_lookup3,num_places = 10))
                       
print(rec_places('fred-wilson', places_train4, user_vecs4, place_vecs4, users12_arr, places12_arr, place_lookup4,num_places = 10))
                       

 
###########################     
#user_vecs, place_vecs = implicit_weighted_ALS(places_train, lambda_val = 0.1, alpha = 15, iterations = 1, rank_size = 20)
                                           
#######Write to CSV file to use cleaner versions for recommendation purpose later#####
#weeplace_data2.to_csv('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkinsMorningCompressed.csv', index=False)'''
print('Done')
print('Execution complete')