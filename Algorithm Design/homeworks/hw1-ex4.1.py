def main():
	#input
	task_pre=[(1,3),(2,6),(5,1),(1,2)]

	#creating dict in order to take keys in preprocessing
	d = { k:v for k,v in task_pre }
    
	#task will be a list of tuples with (time,values sum)
	task = [(t1,sum( v for k,v in task_pre if k == t1 )) for t1 in d.keys() ]
	print("Preprocessed tasks: "+str(task))

	s=2 #daily salary for hired employee h
	S=3 #firing cost for h
	C=2 #hiring cost for h


	#arrays used in the for cycle
	c_s=[]
	c_c=[]

	for i in range(len(task)):
		c_s.append(0)
		c_c.append(0)

	#populating first position of arrays
	c_s[0]=C+s
	c_c[0]=task[0][1]

	for i in range(1,len(task)):
		c_s[i]= min( c_s[i-1] + (task[i][0]-task[i-1][0])*s , c_c[i-1] + C + s, c_s[i-1] + S + C + s )
		c_c[i]= min( c_c[i-1] + task[i][1], c_s[i-1] + S + task[i][1] )

	print "Final value: "+str(min( c_s[i] , c_c[i] ))
	return min( c_s[i] , c_c[i] )

main()