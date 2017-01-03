package edu.iu.RandomForest;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;

public class Leaf extends Node{

    private int majority;

    public Leaf(int majority){
        this.majority = majority;
    }

    public int predict(ArrayList<Integer> a){
        return majority;
    }

	@Override
	public void writeTree(DataOutput out) throws IOException {
		super.writeTree(out);
		out.writeInt(majority);
	}

	@Override
	public int size() {
		return 1;
	}

}
