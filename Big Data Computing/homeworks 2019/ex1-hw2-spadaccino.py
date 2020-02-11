
import os
import sys
from pprint import pprint
from operator import add
from pyspark import SparkContext
from itertools import combinations



def find_common_friends(data_fname, res_dir):
	sc = SparkContext("local[*]", "common_friends")
	text = sc.textFile(data_fname, minPartitions=12)

	# Parse input and produce <nodei, [list of friends of nodei]>
	friends = (text
		.map(lambda line: line.split("\t"))			#split line on tab
		.mapValues(lambda line: line.split(","))	#split friend list on comma
		.filter(lambda x: len(x[1])>=2)				#filter out friend lists with only one element
		.map(lambda x: (int(x[0]), list(map(int, x[1]))))	#convert from str to int
	)

	# Produce pairs of nodes having a common friend <(node1, node2), node_common>
	common_friends = friends.flatMap(
		lambda x: [((min(pair), max(pair)), x[0]) for pair in combinations(x[1], 2)]
	)

	# Aggregate
	result = common_friends.groupByKey().mapValues(list)

	# Save results
	result.saveAsTextFile(res_dir)



def merge_results(res_dir, fname_out):
	"""
	Merge spark part-* files into a correctly-formatted single file
	"""
	tmp_file = "tmp.txt"
	os.system(f"cat {res_dir}/part-* > {tmp_file}")		#merge part files into one file
	f_in, f_out = open(tmp_file), open(fname_out, "w")	#format the single result file
	print("Merging file...")
	for line in f_in:
		t = eval(line)  #laziness
		f_out.write(f"{t[0][0]},{t[0][1]}\t{t[1]}\n")

	f_in.close()
	f_out.close()
	os.remove(tmp_file)



def extract_sample(fame_in, fname_out, n):
	"""
	NOT USED

	Writes on file `fname_out` a sample from `fame_in` taking the first `n` rows.
	The taken sample is consistent with the edge representation: each node which appears in some 
	friend list is going also to appear as row in the output file.
	"""
	assert(n>0)
	data = open(fame_in, "r")
	out = open(fname_out, "w")

	# Read all lines and get which nodes to take
	nodes = set()
	i = 0
	while i < n:
		nodes.add( int(data.readline().strip().split("\t")[0]) )
		i += 1
	
	# Re-read the file taking only the nodes in `nodes` set
	data.seek(0, 0)
	i = 0
	while i < n:
		# Get node and its friend list
		line = data.readline()
		node, friend_list = line.strip().split("\t")
		friend_list = list(map(int, friend_list.split(",")))

		# Get only nodes to write on the sample
		to_write = []
		for friend in friend_list:
			if friend in nodes:
				to_write.append(str(friend))

		if len(to_write) > 0:
			out.write(f"{node}\t{','.join(to_write)}\n")

		i += 1

	data.close()
	out.close()



if __name__ == "__main__":
	data_fname = "soc-LiveJournal1Adj.txt"
	res_dir = "common_friends_result"
	res_merged_fname = "common_friends_result.tsv"

	if os.path.isdir(res_dir):
		print(f"\nFolder {res_dir}/ already on disk, delete it first\n", file=sys.stderr)
		exit(1)

	find_common_friends(data_fname, res_dir)
	merge_results(res_dir, res_merged_fname)
