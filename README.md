# RecSysSocialNetwork

Simple recommendation post service for social network

## Description

We have a social network.  
When registering, students must fill in their profile data, which is stored in the postgres database.  
The social network has a feed that users can scroll through and view random posts from random communities. If you like the post, you can support the author and like.  
All user actions are saved, each of their activity related to viewing posts is also recorded in the database.

## Task

Build a recommendation system for posts on a social network.  
It is necessary to develop a service that will return posts for each user at any time, which will be shown to the user in his social network feed.

## Solution

* Loading data from the database, reviewing data, EDA.
* Creation of features and training sample. content method.
* Training the model and assessing its quality on the validation set.
* Saving the model.
* Writing a service: loading the model -> getting features for the model by user_id -> predicting posts that will be liked -> returning a response.

## Data

Table user_data.  
Contains information about all users of the social network.

| Field name |                                  Overview                                |
|------------|--------------------------------------------------------------------------|
| age        | User age (in profile)                                                    |
| city       | User city (in profile)                                                   |
| country    | User country (in profile)                                                |
| exp_group  | Experimental group: some encrypted category                              |
| gender     | User Gender                                                              |
| id         | Unique user ID                                                           |
| os         | The operating system of the device from which the social network is used |
| source     | Whether the user came to the app from organic traffic or from ads        |

Table post_text_df.  
Contains information about posts and unique ID of each unit with corresponding text and topic.

| Field name |          Overview        |
|------------|--------------------------|
| id         | Unique post ID           |
| text       | Text content of the post |
| topic      | Main theme               |

Table feed_data.  
Contains a history of viewed posts for each user.

| Field name |          Overview        |
|------------|--------------------------|
| timestamp  | The time the viewing was made                                                                                    |
| user_id    | id of the user who viewed                                                                                        |
| post_id    | Viewed post id                                                                                                   |
| action     | Action Type: View or Like                                                                                        |
| target     | Views have 1 if a like was made almost immediately after viewing, otherwise 0. Like actions have a missing value |

## Metric

Hitrate@5

## Requirements

The algorithm should work no more than ~0.5 seconds per 1 request, and take no more than ~4 GB of memory (numbers are approximate).
