# Big-Data-Analytics

Topics: Machine Learning & Human buying behavior

Title: How can Machine Learning help in modeling and predicting human buying behavior for Best Buy?

Problem statement

We will predict which Best Buy product a mobile web visitor will be most interested in based on their
search query or behavior over 2 years.
Questions of interest: Optimizing the mobile app recommendation system in order to increase
incremental sales and potentially streamline supply chain by reducing product offerings based on user
feedback. Can we use past consumer browsing behavior in order to predict future purchases.
Why? Our team’s focus is to gain real world experience on problems companies are having today.
Predicting the next occasion or product selection is a question every business wants to know and being
able to solve this question in a way that is actionable not only helps build experience but brings a skill set
of value to potential employers. By using web data we will be able to find repeated patterns in a users
behavior which will then allow us the ability to identify the next product in their purchase history. An
analysis on web data is also to our team an universal learning we can apply to many other projects.
Motivation: We hope to predict which Best Buy product a mobile web visitor will be most interested in.
Resources used for the project and the dataset ( link to dataset included)
The main data for this project is in the train.csv and test.csv files. These files contain information on what
items users clicked on after making a search.
Link: https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data
Resources:
Best Buy Documentation: https://developer.bestbuy.com/documentation
Best Buy API: https://developer.bestbuy.com/#TableProdRefInfo
and more BigDataR tools: https://github.com/koooee/BigDataR_Examples/tree/master/ACM_comp
https://blog.cloudera.com/blog/2016/05/how-to-build-a-prediction-engine-using-spark-kudu-and-imp
ala/
Other preliminary information:
The analysis was a result of a hackathon in SF in 2014.
The list of tasks each member performed:
Task Team Lead
Data exploration and visualization
● Find open source data set and download
● Explore the data set in terms of its features, attributes (columns),
records (rows), and size, include your findings in your report.
● Visualize the data for a better insight and make it a part of your project
report.
Yajaira Gonzalez
Data Cleanings
● Missing values,
● Duplications,
● Etc.
● include what you needed to do for data leaning in your report
Soren Thomsen
Analysis
● Refer to the problem statement and apply your analysis methods
● Write your queries and illustrate the results, use visualization methods,
and include the analysis explanation into your report.
Lead: Mandeep
Narang
Support:
● Nirlep
● Soren
Thomsen
● Yajaira
Gonzalez
● Sunny Yadav
Conclusion and future work
● make conclusions based on the analysis (part of the report)
● mention potential future work in your report.
Lead: Nirlep
Support: Soren
Thomsen
Report implementation environment
● Computing environment, hardware and software, operating system
● Software tools used in the analysis
● If used, Python version and libraries
● If used, include Python code as an appendix
Lead: Sunny Yadav
Support:
● Yajaira
Gonzalez
● Nirlep
● Soren
Thomsen
● Sunny Yadav
Data Exploration & Visualization
Data set descriptors
Each line of train.csv describes a user's click on a single item and is ( 1865269 rows , 6 columns). It
contains the following fields:
user: A user ID
sku: The stock-keeping unit (item) that the user clicked on
category: The category the sku belongs to
query: The search terms that the user entered
click_time: Time the sku was clicked on
query_time: Time the query was run
Test.csv contains all of the same fields and is ( 1865269 rows , 6 columns). The train.csv includes all of the
same fields except for sku and is( 1865269 rows , 5 columns). We will then estimate which sku's were
clicked on in these test queries. Due to the internal structure of Best Buy's databases, there is no
guarantee that the user clicks resulted from a search with the given query. What we do know is that the
user made a query at query_time, and then, at click_time, they clicked on the sku, but we don't know that
the click came from the search results. The click_time is never more than five minutes after the
query_time. In addition, there is information about products, product categories, and product reviews in
product_data.tar.gz.
Data Cleaning Process:
A large part of our initial analysis was focused on getting the data converted from .xml files to csv files.
Once this was completed we removed a portion of the data that we did not need for analysis.
Visualizations:
Understand Rating dataset for user-user similarity model (elaborate more on our EDA and why we looked
at certain things)
This title shows the frequency of product reviews and clearly
shows that most products on the mobile app only have one or
two reviews which limited the scope of our analysis
Most product reviews show mainly 5 star ratings
Here the density of Ratings is shown. Where we can
see how density of users who have given a rating 1 for
a product differs from density of users who have given
a rating of 2, 3, 4 and 5
Here we see the frequency of votes for each ratings i.e 1, 2, 3, 4, 5
We see that most users have rated a product only once.
Title length exploration
Here we can see that there is no direct
obvious relation between the Title
length of a product and the ratings
given by user
Comment length exploration
Even here we can see that there is no
direct obvious relation between the
Title length of a product and the
ratings given by user
Rating: 1 Rating: 2
Rating: 3 Rating: 4
Rating: 5
Here is the word cloud of all ratings and the size of each word indicates
its frequency or importance. That is to see what words were more
frequently used to rate a certain product.
Analysis
To help predict the best product for a consumer based on past online history we must create a
recommendation model. Our research pointed us to 3 implementable strategies based on existing best
practices for recommendation models. Below is a summary of the queries and methods we employed for
each strategy.
Data Preparation:
To prepare our data for the analysis we filtered products that were unpopular and focused on the ‘sku’,
‘user’, ‘name’, ‘flag’ resulting in a ‘merged’ data frame of 329148 rows, 5 columns. We use this df in our
analysis below to cross reference recommendations with product name, category based on user & sku.
Collaborative Filtering:
A method of making automatic predictions (filtering) about the interests of a user by collecting preferences
or taste information from many users (collaborating). There are two categories of CF, user-based &
item-based.
Summary: This was an unsupervised model based on two frameworks, pearson correlation and cosine
similarity. We dropped the users from the matrix and used item/user columns in order to check the
relations between each item/user column. We then sort each items/user based on correlations coefficients
in order to find similarities and take top n items to be used for recommending to new users. Generally we
used between 10 and 20 items.
The first part of the CF method is to create a matrix.
Steps in Matrix development:
1. Create User & Item Sparse matrix; User set as index and product sku set as columns (for user based
we used users for the columns), the values in the matrix belong to existing ratings
R_df.shape
271,937 rows x 976 columns
Understanding the quality of the matrix:
User/Item Based, Using Sparse matrix - Using Pearson Correlation
User similarity works well if number of items is much smaller than the number of users and if the items
change frequently. Item similarity works best when user base is smaller.
For our user and item based CF model we looked at similarity by employing the use of Pearson’s
correlation coefficient. To help illustrate how Person correlation works, let be the r u,i rating of the ith
item under user 𝑢, r u be the average rating of user 𝑢, and 𝑐𝑜𝑚(𝑢, 𝑣) be the set of items rated by both user
𝑢 and user 𝑣. The similarity between user u and user v is given by Pearson’s correlation coefficient.
For our analysis and application of PC in our python code we first selected 2 users. We then defined the
PC function to find correlations of user 1 & 2 based on existing ratings that matched those users behavior
in the developed matrix. The table below shows what users are closely related to user 1. Please reference
code for item correlation table.
.
This information from the PC table of correlations then helped take our analysis a step further by being
able to cross reference the full ‘merged’ data frame and understand what products have been already
viewed, viewed by others similar to the user that are unique and provide a list of recommended items. The
code below helped generate the more detailed recommendation output.
Final Recommendation from User based, Sparse Matrix using PC:
(Based only on User1)
User Based, Using Sparse matrix - Using Cosine Similarity
Same approach was produced for the item-based category
After assessing our recommendation based off of PC, we wanted to take a look at different methods,
focusing on cosine similarity in this section. Cosine similarity is a measure of similarity between two
non-zero vectors of an inner product space that measures the cosine of the angle between them.
Step 1. Create a copy of the ratings matrix with all null values imputed as 0; r_matrix_dummy
Step 2.. Create a matrix based off of user 1 & 2, we named this user_rat in our code. [2 rows x 976
columns]
Step 3. Compute cosine similarity matrix using the ratings matrix and the user matrix
Step 4. Review qualities of the matrix
Now that we have created this matrix, we can edit the table to showcase the user by user
recommendation. In the table below you can see we only looked at user 1 and we ranked the similarity of
users for better viewability.
This information is now ready to be run into our recommendation function for better understanding of what
the cosine similarity approach produced for user 1.
Model Based Collaborative Filtering
Based on the item & user data set, a utility matrix was composed using SVDpp, SVD and NMF found in
the python Surprise library. This is a library that was created specifically for collaborative filtering of data.
We then computed similar browsing trends between users by comparing rows of the user matrix.
Accuracy of the model is based on the RMSE scores after running model on the test set.
Matrix Factorization-based algorithms
Class of collaborative filtering algorithms used in recommender systems. The essence is that MF helps
represent users & items in a lower dimensional latent space
SVD: algorithm is equivalent to Probabilistic Matrix Factorization
● Decomposes a matrix A into the best lower rank (i.e. smaller/simpler) approximation of the
original matrix A. To get the lower rank approximation, we take these matrices and only
keep the top k features, which can be thought of as the underlying tastes and preferences
vectors.
SVD++: algorithm is an extension of SVD that takes into account implicit ratings.
NMF: collaborative filtering algorithm based on Non-negative Matrix Factorization. It is very similar
with SVD.
The data we are using is the merged data frame we created in our data prep with the columns ‘user’,
’name’, ‘flag’. Below is the code we implemented to run these algorithms and understand the RMSE.
Based on these results we opted to proceed further with the SVD++ algorithm.
We then used the scikit-learn library to split the dataset into testing and training.
Cross_validation.train_test_split shuffles and splits the data into two datasets according to the percentage
of test examples, which in this case is 0.25. We will also use the accuracy metric of rmse. We’ll then use
the fit() method which will train the algorithm on the train set, and the test() method which will return the
predictions made from the testset
From this we can see that the model is generalized and the rmse score is also good.
In our code we conducted a deeper dive of our worst and best predictors to further inspect results. For
additional details please reference line 108-109.
The new predictions code following this new matrix is below:
For his method when we look for user1 we see the recommendation below
Collaborative Filtering using ALS on pySpark
Alternating least squares model:
Supervised model implemented in Apache Spark traditionally used for large scale collaborative filtering
problems. The model works by holding the user matrix fixed and then running gradient descent with the
user matrix. We chose to implement this model because it helped us solve issues with scalability as well
as the spareness of the ratings data we uncovered within the data set. We also chose to evaluate this
model using RMSE.
From these results we used the best parameter ( Root-mean-square error = 0.5590686336163447) to generate
recommendations for each user. Below is the table of recommendations once we applied the code.
To reiterate the steps we followed can be broken out below and can be observed in our code from the
matrix factorization method utilizing the ALS.
Step 1. Data Splitting: Stratified Sampling
Step 2. Stochastic Gradient Descent; Starting with random numbers
Step 3. Predicting and optimizing/minimizing RMSE for already present ratings/flag in the matrix on
Training Data
Step. 4 Prediction for the Test Data; can be seen from the resulting matrix of predictions
Conclusion & Future Work
Due to the challenges within the data sets provided we conducted several different types of analysis to
build out our recommendation system. The following was used, collaborative filtering, using item and
user-based similarity with the use of cosine similarity and pearson correlation coefficients. The use of
model collaborative filtering, using SVDpp, SVD and NMF. Lastly collaborative filtering with MF specific to
ALS. What we learned is that we cannot rely on matrix factorization.
Insights of Matrix created:
Due to the size of zero occurrences in the Best Buy dataset we moved on from implementing a specific
modeling CF strategy. In an ideal situation we would advise any analysis using matrix factorization to have
observable values for more than 25% of the data. We only have flags in our data, not any true ratings,
therefore these, binary recommendation models can’t work.
Our approach would be to deploy the model and capture users response which will then lead us to update
and optimize the model, we cannot rely on just numbers we need to rely on the users response to help us
flesh out a fully working model.
Implementation Platform
Stack
● Python
● Spark
● Etc
Modeling/Machine Learning
● Scikit-learn
● Pandas
● Numpy
● Spark ML - Lib
● Surprise
● Thinkstat
● Thinkplot
Data Visualization
● Plotly
● Interactive
● Matplotlib
● Seaborn
