import mpi.MPI;

import java.util.*;
import java.io.*;

public class MPIPR {
	//global rankValues HashMap initializes each node's rank to 1/n
	private HashMap<Integer, Double> rankValues = new HashMap<Integer, Double>();

	//global adjacencyMatrix/outBoundNodes HashMap and updates it with dangling nodes
	private HashMap<Integer, ArrayList<Integer>> adjacencyMatrix = new HashMap<Integer, ArrayList<Integer>>();

	//global outBoundNodesCount HashMap and updates it with dangling nodes count
	private HashMap<Integer, Integer> outBoundNodesCount = new HashMap<Integer, Integer>();

	//global inBoundNodes HashMap based on outBoundNodes HashMap
	private HashMap<Integer, ArrayList<Integer>> inBoundNodes = new HashMap<Integer, ArrayList<Integer>>();

	private int iterations, sizeOfInput;
	private static int totalNumUrls;
	private String inputFile, outputFile;

	private double df;
	public static void main(String[] args) throws InterruptedException, IOException{

		MPI.Init(args);
		MPIPR mpipr = new MPIPR();

		int rank = MPI.COMM_WORLD.Rank();		//ranks are assigned to each process
		int nproc = MPI.COMM_WORLD.Size();		//there are nproc number of processes

		//parse the arguments
		mpipr.parseArgs(args);

		//read the file into adjmatrx HashMap
		mpipr.loadInput();
		
		//System.out.println("after:" + mpipr.adjacencyMatrix.size());;
		int chunkSize= mpipr.adjacencyMatrix.size()/(nproc-1);
		int remainingChunks = mpipr.adjacencyMatrix.size() % chunkSize;



		double[] chunk;
		int chunkSizeActual = 0;
		if(rank == 0){
			float numOfUrls;
			numOfUrls = Collections.max(mpipr.adjacencyMatrix.keySet());
			Double initializePageRank = (double) (1 / (numOfUrls+1));
			for (Integer key : mpipr.adjacencyMatrix.keySet()){
				mpipr.rankValues.put(key,initializePageRank);
				if(mpipr.adjacencyMatrix.get(key).size()==0){
					ArrayList<Integer> list = new ArrayList<Integer>();
					for(int i=0;i<numOfUrls;i++){
						if (i==key){
							continue;
						}
						else{
							list.add(i);
						}
					}
					mpipr.adjacencyMatrix.put(key,list);
				}
			}
			
			System.out.println("Starting chunks");
			//for every process
			for(int i = 1; i < nproc; i++){
				//splits the rankValues into n/nproc chunks and adds remaining nodes to the last process itself

				if(i == nproc){
					chunkSizeActual = chunkSize + remainingChunks;
					chunk = mpipr.getChunk(i - 1, chunkSizeActual);
				}
				else{
					chunkSizeActual = chunkSize;
					chunk = mpipr.getChunk(i - 1, chunkSizeActual);
				}
				//send the chunks
				MPI.COMM_WORLD.Send(chunk, 0, chunk.length, MPI.DOUBLE, i, 1);
			}
			//for every iteration,
			for(int iter =0;iter < mpipr.iterations; iter++){
				//wait for all processes to send back the calculated pageRank for their nodes
				System.out.println("iteration:" + iter);
				int receivedRanks[] = new int[chunkSize + remainingChunks];
				MPI.COMM_WORLD.Recv(receivedRanks, 0, chunkSizeActual, MPI.DOUBLE, rank, 1);
				//MPI.COMM_WORLD.wait();

				//combine the output of all processes into the global rankValues HashMap
				double[] tempBuf = new double[mpipr.adjacencyMatrix.size()];
				double[] op = new double[chunkSizeActual];
			    for (int j = 1; j <= chunkSizeActual; j *= 10)  {
			      for (int i = 0; i < j; i++) {
			    	  op[i] = rank;
			      }
			    }
				MPI.COMM_WORLD.Allgather(op, 0, chunkSizeActual, MPI.DOUBLE, tempBuf, 0, mpipr.rankValues.size(), MPI.DOUBLE);
			    //MPI.COMM_WORLD.Allreduce(sendbuf, sendoffset, recvbuf, recvoffset, count, datatype, op);

				
				for(int i = 1; i < nproc; i++){
					if(i == nproc)
						chunk = mpipr.getChunk((i - 1) * chunkSize, chunkSize + remainingChunks);
					else
						chunk = mpipr.getChunk((i - 1) * chunkSize, chunkSize);

					//send the chunks
					MPI.COMM_WORLD.Send(chunk, 0, chunkSizeActual, MPI.DOUBLE, i, 1);
				}
			}

			//after iteration completes, print the rankValues
			mpipr.printValues();
		}
		else{
			//sub
			System.out.println("chunkSize:"+chunkSize + "reamining:" +remainingChunks);
			double localRanks[] = new double[chunkSize + remainingChunks];
			//receives the chunk of nodes of which the rank values have to be calculated
			MPI.COMM_WORLD.Recv(localRanks, 0, chunkSizeActual, MPI.DOUBLE, rank, 1);

			//calculate the pagerank
			localRanks = mpipr.calculatePageRank(chunkSizeActual);

			//send the new pageRank values chunk back to the main process
			MPI.COMM_WORLD.Send(localRanks, 0, chunkSizeActual, MPI.DOUBLE, rank, 1);
		}
	}
	//for every node in the chunk, calculates pageRank based on the global outBoundNodesCount and inBoundNodes HashMaps
	private double[] calculatePageRank(int chunkSize) {
		float numOfUrls = Collections.max(this.adjacencyMatrix.keySet());
		double[] tempRanks = new double[chunkSize];
		for (int page=0; page<=numOfUrls; page++){
			Double dampFact = ((1-this.df)/numOfUrls);
			Double dPageRank = 0.0;
			for(Integer key : this.inBoundNodes.keySet()){
				if(this.inBoundNodes.get(key).contains(page)) {
					//dPageRank += (this.rankValues.get(key) / this.adjacencyMatrix.get(key).size());
					dPageRank += (this.rankValues.get(key) / this.outBoundNodesCount.get(key));
				}
			}
			dPageRank = dampFact + (this.df * dPageRank);
			//this.rankValues.put(page,intPageRank);
			tempRanks[page] = dPageRank;
		}
		/*
		try{
			PrintWriter writer = new PrintWriter(this.outputFile);
			for (int page=0; page<=numOfUrls; page++){
				writer.println(page+ "\t" + this.rankValues.get(page));
			}
			writer.close();
		}
		catch (IOException ex) {
			System.out.println("File Error");
		}*/
		return tempRanks;
	}

	private double[] getChunk(int startIndex, int inpSize) {
		double[] retArr = new double[inpSize];
		for(int c = startIndex; c < inpSize; c++){
			retArr[c] = this.rankValues.get(c);
		}
		return retArr;
	}

	private void printValues() {
		List <Double> listPR = new ArrayList<Double>(this.rankValues.values());
		Collections.sort(listPR);
		Collections.reverse(listPR);
		listPR = listPR.subList(0, 10);
		System.out.println("Number of Iterations : " + this.iterations);
		for(Double key: listPR){
			for (Map.Entry<Integer, Double> e : this.rankValues.entrySet()) {
				if(e.getValue()==key){
					System.out.println("Page:" + e.getKey() + "\t\t" + "PageRank:"+key);
				}
			}
		}
	}

	private void loadInput() throws IOException {
		FileReader read = new FileReader(new File(this.inputFile));
		BufferedReader buffRead = new BufferedReader(read);
		String ln;
		while((ln = buffRead.readLine())!=null){
			Scanner nodes = new Scanner(ln);
			ArrayList<Integer> ref = new ArrayList<Integer>();
			// pull out the first node to form the hashMap key
			int node = nodes.nextInt();
			// pull out the subsequent nodes to form the hashMap values
			while(nodes.hasNextInt()){
				ref.add(nodes.nextInt());
			}
			this.adjacencyMatrix.put(node,ref);
			nodes.close();
		}
		//System.out.println("adj size:"+ this.adjacencyMatrix.size());
		read.close();
		
		/*while((ln = buffRead.readLine())!=null){
				Scanner nodes = new Scanner(ln);
				ArrayList<Integer> ref = new ArrayList<Integer>();
				// pull out the first node to form the hashMap key
				int node = nodes.nextInt();
				// pull out the subsequent nodes to form the hashMap values
				while(nodes.hasNextInt()){
						ref.add(nodes.nextInt());
				}
				this.adjacencyMatrix.put(node,ref);
				nodes.close();
		}*/

		for (Integer key : this.adjacencyMatrix.keySet()){
				int num = 0;
				if(this.adjacencyMatrix.get(key).size()==0){
					num = this.adjacencyMatrix.keySet().size()-1;
				}
				else{
					num = this.adjacencyMatrix.get(key).size();
				}
				this.outBoundNodesCount.put(key,num);
		}

		for (Integer k : this.adjacencyMatrix.keySet()){
				ArrayList<Integer> ref = new ArrayList<Integer>();
				for (Integer key : this.adjacencyMatrix.keySet()){
					if(this.adjacencyMatrix.get(key).contains(k)){
							ref.add(key);
					}
				}
				this.inBoundNodes.put(k,ref);
		}

		totalNumUrls = Collections.max(this.adjacencyMatrix.keySet());
		Double initializePageRank = (double) (1 / (totalNumUrls+1));
		
		for (Integer key : this.adjacencyMatrix.keySet()){
				this.rankValues.put(key,initializePageRank);
		}
		
//		int chunkSize = totalNumUrls / sizeOfInput;
//		int local_chunkSize = 0;
//		int sizeBuf[] = new int[chunkSize];
//		ArrayList<Integer> listOfKeys = new ArrayList<Integer>(this.adjacencyMatrix.keySet());
//		System.out.println(listOfKeys);
	}

	private void parseArgs(String[] args) {
		try{
		/*for(int h =0;h<7;h++)
			System.out.println(args[h]);
		System.out.println("-------------");*/
		
		
		this.inputFile = args[3];
		//System.out.println(this.inputFile);
		this.outputFile = args[4];
		//System.out.println(this.outputFile);
		this.iterations = Integer.parseInt(args[5]);
		//System.out.println(this.iterations);
		this.df = Double.parseDouble(args[6]);
		//System.out.println(this.df);
		}
		catch (Exception ex){
			System.out.println(ex.getMessage());
		}
		
	}
}