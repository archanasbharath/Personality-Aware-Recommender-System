// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.personality;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import happy.coding.io.FileIO;
import librec.data.DenseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.SocialRecommender;

/**
 * Hao Ma, Dengyong Zhou, Chao Liu, Michael R. Lyu and Irwin King,
 * <strong>Recommender systems with social regularization</strong>, WSDM
 * 2011.<br>
 * 
 * <p>
 * In the original paper, this method is named as "SR2_pcc". For consistency, we
 * rename it as "SoReg" as used by some other papers such as: Tang et al.,
 * <strong>Exploiting Local and Global Social Context for
 * Recommendation</strong>, IJCAI 2013.
 * </p>
 * 
 * @author guoguibing
 * 
 */
public class PersReg1 extends SocialRecommender {

	private Table<Integer, Integer, Double> userCorrs;
	private Table<Integer, Integer, Double> persCorrs;
	private float beta;
	private String perssim;
	private String yes = "Y";
	private Map<Integer, double[]> persMap;

	public PersReg1(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();
		readData();
		userCorrs = HashBasedTable.create();
		persCorrs = HashBasedTable.create();
		beta = algoOptions.getFloat("-beta");
		perssim = algoOptions.getString("-perssim");
		System.out.println("soc-rows:"+socialMatrix.numRows());
		System.out.println("soc-cols:"+socialMatrix.numColumns());
		System.out.println("train-rows:"+trainMatrix.numRows());
		System.out.println("train-cols:"+trainMatrix.numColumns());
	}

	/**
	 * compute rating based similarity between users u and v
	 */
	protected double similarity(Integer u, Integer v) {
		if (userCorrs.contains(u, v))
			return userCorrs.get(u, v);

		if (userCorrs.contains(v, u))
			return userCorrs.get(v, u);

		double sim = Double.NaN;

		if (u < trainMatrix.numRows() && v < trainMatrix.numRows()) {
			SparseVector uv = trainMatrix.row(u);
			if (uv.getCount() > 0) {
				SparseVector vv = trainMatrix.row(v);
				sim = correlation(uv, vv, "pcc"); // could change to other measures

				if (!Double.isNaN(sim))
					sim = (1.0 + sim) / 2;
			}
		}

		userCorrs.put(u, v, sim);

		return sim;
	}

	/**
	 * compute personality based similarity between users u and v
	 */
	protected double perssimilarity(Integer u, Integer v) {
		
		if (persCorrs.contains(u, v))
			return persCorrs.get(u, v);

		if (persCorrs.contains(v, u))
			return persCorrs.get(v, u);

		double sim = Double.NaN;
		int persmapSize = persMap.size();
	
		if (u < persmapSize && v < persmapSize) {
			SparseVector iv = new SparseVector(persMap.get(u).length, persMap.get(u));
			if (iv.getCount() > 0) {

				SparseVector jv = new SparseVector(persMap.get(v).length, persMap.get(v));

				sim = correlation(iv, jv, "pcc");

				if (!Double.isNaN(sim))
					sim = (1.0 + sim) / 2;
			}
		}
				persCorrs.put(u, v, sim);
			

		return sim;
	}

	@Override
	protected void buildModel() throws Exception {
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;

			// temp data
			DenseMatrix PS = new DenseMatrix(numUsers, numFactors);
			DenseMatrix QS = new DenseMatrix(numItems, numFactors);

			// ratings
			for (MatrixEntry me : trainMatrix) {
				int u = me.row();
				int j = me.column();
				double ruj = me.get();
				double iuj = 0;
				if (ruj >= 1) {
					iuj = 1;
				}else {
					iuj = 0;
				}

				double pred = predict(u, j);
				double euj = pred - ruj;

				loss += iuj * euj * euj;

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);

					PS.add(u, f, euj * qjf + regU * puf);
					QS.add(j, f, euj * puf + regI * qjf);

					loss += regU * puf * puf + regI * qjf * qjf;
				}
			}

			
			// friends
			for (int u = 0; u < numUsers; u++) {
				// out links: F+
				SparseVector uos = socialMatrix.row(u);
				
				for (int k : uos.getIndex()) {
					double suk = 0.0;
					if (perssim.equalsIgnoreCase(yes)) {
						suk = perssimilarity(u, k);
					}else {
						suk = similarity(u,k);
					}
					if (!Double.isNaN(suk)) {
						for (int f = 0; f < numFactors; f++) {
							double euk = P.get(u, f) - P.get(k, f);
							PS.add(u, f, beta * suk * euk);
							loss += beta * suk * euk * euk;
						}
					}
				}

				/*
				// in links: F-
				SparseVector uis = socialMatrix.column(u);
				
				//System.out.println("soc-mat-col:"+uis.size());
				for (int g : uis.getIndex()) {
					
					double sug = 0.0;
					if (perssim.equalsIgnoreCase(yes)) {
						sug = perssimilarity(u, g);
					}else {
						sug = similarity(u,g);
					}
					if (!Double.isNaN(sug)) {
						for (int f = 0; f < numFactors; f++) {
							double eug = P.get(u, f) - P.get(g, f);
							PS.add(u, f, beta * sug * eug);
						}
					}
				}
				*/

			} // end of for loop
			

			P = P.add(PS.scale(-lRate));
			Q = Q.add(QS.scale(-lRate));

			loss *= 0.5;

			if (isConverged(iter))
				break;
		}
	}

	/**
	 * Read the users personality data and store them in a hashmap with userid as
	 * the key and the personality dimensions acquired by FFM as the values
	 */
	private void readData() {
		try {

			BufferedReader br = FileIO.getReader(cf.getPath("dataset.personality"));
			persMap = new HashMap<Integer, double[]>();

			String line = null;
			while ((line = br.readLine()) != null) {
				String[] data = line.split("[ ,]");

				Integer userId = Integer.parseInt(data[0]);

				double[] persDim = { Double.parseDouble(data[5]), Double.parseDouble(data[6]),
						Double.parseDouble(data[7]), Double.parseDouble(data[8]), Double.parseDouble(data[9]) };

				int userInnerId = rateDao.getUserId(userId.toString());
				persMap.put(userInnerId, persDim);

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException iex) {
			iex.printStackTrace();
		}

	}

	
}
