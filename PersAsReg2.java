package librec.personality;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.intf.IterativeRecommender;
//Copyright (C) 2014 Guibing Guo
//
//This file is part of LibRec.
//
//LibRec is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//LibRec is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

/**
 * Biased Matrix Factorization Models. <br/>
 * 
 * NOTE: To have more control on learning, you can add additional regularation
 * parameters to user/item biases. For simplicity, we do not do this.
 * 
 * @author guoguibing
 * 
 */
public class PersAsReg2 extends IterativeRecommender {

	
	private DenseMatrix A;

	private Map<Integer, double[]> persMap;

	private Map<Integer, Double> nns;

	private DenseVector itemMeans;

	private SymmMatrix persCorrs;

	public PersAsReg2(SparseMatrix rm, SparseMatrix tm, int fold) {
		super(rm, tm, fold);
	}

	protected void initModel() throws Exception {

		super.initModel();
		readData();
		buildPersCorrs();

		knn = algoOptions.getInt("-knn");

		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		A = new DenseMatrix(numUsers, numFactors);

		// initialize user bias
		userBias.init(initMean, initStd);
		itemBias.init(initMean, initStd);
		A.init(initMean, initStd);

		itemMeans = new DenseVector(numItems);

		for (int j = 0; j < numItems; j++) {
			SparseVector h = trainMatrix.column(j);
			itemMeans.set(j, h.getCount() > 0 ? h.mean() : globalMean);
		}

	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;

			for (MatrixEntry me : trainMatrix) {

				int u = me.row(); // user
				int j = me.column(); // item

				double ruj = me.get();

				double pred = predict(u, j);
				double euj = ruj - pred;

				loss += euj * euj;

				// update bias
				double bu = userBias.get(u);
				double sgd = euj - regB * bu;
				userBias.add(u, lRate * sgd);

				loss += regB * bu * bu;

				// update item bias
				double bi = itemBias.get(j);
				double sgdi = euj - regB * bi;
				itemBias.add(j, lRate * sgdi);

				loss += regB * bi * bi;

				findNN(u, j);
				double simProd = 0;
				double simSum = 0;
				double simDiv = 0;
				for (Entry<Integer, Double> en : nns.entrySet()) {
					int nn = en.getKey();
					double simP = en.getValue();
					simSum += simP;
					for (int f = 0; f < numFactors; f++) {
						double pa = A.get(nn, f);
						simProd = simProd + (simP * pa);
					}
				}
				
				simDiv = simProd/simSum;
				for (Entry<Integer, Double> en : nns.entrySet()) {
					int nn = en.getKey();
					for (int f = 0; f < numFactors; f++) {
						double pu = P.get(u, f);
						A.add(nn, f, 2 * lRate * regU * ( pu - simDiv));
					}
				}
				
				
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);

					double delta_u = euj * qjf - regU * puf - regU * ( puf - simDiv);
					double delta_j = (euj * puf) - (regI * qjf);

					P.add(u, f, lRate * delta_u);

					Q.add(j, f, lRate * delta_j);

					loss += regU * puf * puf + regI * qjf * qjf + regU * (puf - simDiv) * (puf - simDiv);

				}
			}

			loss *= 0.5;

			if (isConverged(iter))
				break;

		} // end of training

	}

	/**
	 * predict the rating of item j by user u
	 */
	protected double predict(int u, int j) throws Exception {

		return itemMeans.get(j) + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
	}

	public double dotProd(DenseMatrix m, int mrow, DenseMatrix n, int nrow) {
		int mcols = m.numColumns();
		int ncols = n.numColumns();
		assert mcols == ncols;

		double res = 0;
		for (int fac = 0; fac < mcols; fac++) {
			double persPlusUserFactor = 0.0;
			persPlusUserFactor = m.get(mrow, fac);
			res += persPlusUserFactor * n.get(nrow, fac);
		}
		return res;
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

	/*
	 * Find nearest neighbors for the given user
	 */
	public void findNN(int u, int j) {
		// find a number of similar users
		nns = new HashMap<>();
		SparseVector dv = persCorrs.row(u); // get similarities of given user Vs all other users

		for (int v : dv.getIndex()) {
			double sim = dv.get(v);
			double rate = trainMatrix.get(v, j);

			if (isRankingPred && rate > 0)
				nns.put(v, sim); // similarity could be negative for item ranking
			else if (sim > 0 && rate > 0)
				nns.put(v, sim);
		}

		// topN similar users
		if (knn > 0 && knn < nns.size()) {
			List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true); // sorting nns
			List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn); // subsetting just top knn
			nns.clear(); // clearing the old nns that contains all neighbors similarities
			for (Map.Entry<Integer, Double> kv : subset) // loading nns now with just knn similarities
				nns.put(kv.getKey(), kv.getValue());
		}
	}

	/*
	 * build personality based similarity matrix for every user Vs the other users
	 */

	public void buildPersCorrs() {

		persCorrs = new SymmMatrix(persMap.size());

		for (int row = 0; row < persMap.size(); row++) {
			SparseVector iv = new SparseVector(persMap.get(row).length, persMap.get(row));

			for (int col = row + 1; col < persMap.size(); col++) {
				SparseVector jv = new SparseVector(persMap.get(col).length, persMap.get(col));

				double sim = correlation(iv, jv);

				if (!Double.isNaN(sim)) {

					persCorrs.set(row, col, sim);

				}
			}
		}

	}

}// class ends
