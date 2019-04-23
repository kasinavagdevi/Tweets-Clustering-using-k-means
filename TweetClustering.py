import json
import sys
import re, string
import numpy as np

from nltk.corpus import stopwords

regex = re.compile('[%s]' % re.escape(string.punctuation))
stop_words_list = set(stopwords.words('english'))


def cluster_tweets(k, tweets_data, seeds_file, total_iterations):
    tweets = dict()
    with open(tweets_data, 'r') as tweetsFile:
        for tweet in tweetsFile:
            line = json.loads(tweet)
            tweets[line['id']] = line

    seeds = list()
    with open(seeds_file, 'r') as seedsfile:
        for tweet in seedsfile:
            line = int(tweet.rstrip(',\n'))
            seeds.append(line)

    clusters = {}  # cluster to tweetID
    id_clusters = {}  # tweetID to cluster
    # Initialize tweets to no cluster
    for tweet in tweets:
        id_clusters[tweet] = -1

    # Initialize clusters with seeds
    for cluster_no in range(k):
        clusters[cluster_no] = {seeds[cluster_no]}
        id_clusters[seeds[cluster_no]] = cluster_no

    jaccard_dist_matrix = calculate_jaccard_matrix(tweets)  # jaccard distance in a matrix
    new_clusters_id, new_id_clusters = build_new_clusters(tweets, clusters, id_clusters, k, jaccard_dist_matrix)
    clusters = new_clusters_id
    id_clusters = new_id_clusters

    iterations = 1
    while iterations < total_iterations:
        new_clusters_id, new_id_clusters = build_new_clusters(tweets, clusters, id_clusters, k, jaccard_dist_matrix)
        if id_clusters == new_id_clusters:  # no change in clusters
            break
        iterations += 1
        clusters = new_clusters_id
        id_clusters = new_id_clusters
    with open('out.txt', 'w') as output:
        for cluster_no in clusters:
            # print(str(cluster_no) + "\n" + '          ' + ','.join(map(str, clusters[cluster_no])))
            output.write(str(cluster_no) + "       " + ','.join(map(str, clusters[cluster_no]))+'\n')


def calculate_jaccard_matrix(tweets):
    jaccard_dist_matrix = {}
    for i in tweets:
        jaccard_dist_matrix[i] = {}
        setA = pre_process(tweets[i]['text'])
        for j in tweets:
            if j not in jaccard_dist_matrix:
                jaccard_dist_matrix[j] = {}
            setB = pre_process(tweets[j]['text'])
            distance = jaccardDistance(setA, setB)
            jaccard_dist_matrix[i][j] = distance
            jaccard_dist_matrix[j][i] = distance
    return jaccard_dist_matrix


def build_new_clusters(tweets, clusters_id, id_clusters, k, jaccard_dist_matrix):
    new_clusters_id = {}
    new_id_clusters = {}
    for i in range(k):
        new_clusters_id[i] = set()
    for i in tweets:
        min_distance = float("inf")
        min_cluster = id_clusters[i]

        # Calculate min average distance to each cluster
        for j in clusters_id:
            dist = 0
            count = len(clusters_id[j])
            for l in clusters_id[j]:
                dist += jaccard_dist_matrix[i][l]
            if count > 0 and (dist / float(count)) < min_distance:
                min_distance = dist / float(count)
                min_cluster = j
        new_clusters_id[min_cluster].add(i)
        new_id_clusters[i] = min_cluster
    return new_clusters_id, new_id_clusters


def pre_process(line):
    words = line.lower().strip().split(' ')
    to_return = set()
    for word in words:
        word = word.rstrip().lstrip()
        # print(word)
        # if not (re.match(r'^https?:\/\/.*[\r\n]*', word)) and not (re.match('^@.*', word)) and
        # not (re.match('\s', word)) and word not in (stopWordsList) and (word != 'rt') and (word != ''):
        if (not (re.match(r'^https?://.*[\r\n]*', word))) and \
                (not (re.match('^@.*', word))) and word not in stop_words_list and not (re.match('rt', word)):
            to_return.add(regex.sub('', word))
    return to_return


# Jaccard Distance
def jaccardDistance(A, B):
    try:
        return 1 - float(len(A.intersection(B))) / float(len(A.union(B)))
    except ZeroDivisionError:
        print("ERROR")


# Main function to take the arguments.
if __name__ == "__main__":
    # As there should be atleast 3 arguments.
    if (len(sys.argv)) < 3:
        print("Not a valid argument list")
    else:
        total_clusters = int(sys.argv[1])  # 25 is the fixed number of clusters
        seed_data_file = sys.argv[2]  # We give the seed file
        file_name = sys.argv[3]  # takes the tweet data file.
        cluster_tweets(total_clusters, file_name, seed_data_file, 1000)
