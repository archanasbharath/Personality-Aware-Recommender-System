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

package librec.personality;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import happy.coding.io.Strings;
import happy.coding.math.Stats;
import librec.data.Configuration;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.intf.Recommender;

/**
 * <h3>User-based Nearest Neighbors</h3>
 * 
 * <p>
 * It supports both recommendation tasks: (1) rating prediction; and (2) item
 * ranking (by configuring {@code item.ranking=on} in the librec.conf). For item
 * ranking, the returned score is the summation of the similarities of nearest
 * neighbors.
 * </p>
 * 
 * <p>
 * When the number of users is extremely large which makes it memory intensive
 * to store/precompute all user-user correlations, a trick presented by (Jahrer
 * and Toscher, Collaborative Filtering Ensemble, JMLR 2012) can be applied.
 * Specifically, we can use a basic SVD model to obtain user-feature vectors,
 * and then user-user correlations can be computed by Eqs (17, 15).
 * </p>
 * 
 * @author guoguibing
 * 
 */
@Configuration("knn, similarity, shrinkage")
public class PKNN2 extends Recommender {

	// user: nearest neighborhood
	private String perssim;
	private String yes = "Y";
	private double m; 
	private SymmMatrix ratCorrs;
	private SymmMatrix persCorrs;
	private DenseVector userMeans;
	private DenseVector itemMeans;

	private Map<Integer, double[]> persMap;

	public PKNN2(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	@Override
	protected void initModel() throws Exception {

		readData();

		buildRatCorrs();

		buildPersCorrs();

		perssim = algoOptions.getString("-perssim");
		m = algoOptions.getFloat("-m");
		
		userMeans = new DenseVector(numUsers);

		for (int u = 0; u < numUsers; u++) {
			SparseVector uv = trainMatrix.row(u);
			userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
		}
		
		itemMeans = new DenseVector(numItems);
		
		for (int j = 0 ; j <numItems; j++) {
			SparseVector h = trainMatrix.column(j);
			itemMeans.set(j,  h.getCount() > 0 ? h.mean():globalMean);
		}
		
	}

	@Override
	protected double predict(int u, int j) {
		
		SparseVector dv;

		// find a number of similar users
		Map<Integer, Double> nns = new HashMap<>();
		if (perssim.equalsIgnoreCase(yes)) {
			dv = persCorrs.row(u); // use personality similarity
		}else {
			dv = ratCorrs.row(u); // use rating similarity
		}
		
		//

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
			List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
			List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
			nns.clear();
			for (Map.Entry<Integer, Double> kv : subset)
				nns.put(kv.getKey(), kv.getValue());
		}

		if (nns.size() == 0)
			return isRankingPred ? 0 : globalMean;

		if (isRankingPred) {
			// for item ranking

			return Stats.sum(nns.values());
		} else {
			// for rating prediction
			double sum = 0, nnAverageRating = 0;
			
			for ( Entry<Integer, Double> en : nns.entrySet()) {
				int v = en.getKey();
				double rate = trainMatrix.get(v, j);
				sum += rate;
			}
			
			nnAverageRating = sum/nns.size();
			double predRating = ((nns.size()/(nns.size() + m)) * nnAverageRating ) + ((m/(nns.size() + m)) * itemMeans.get(j));
			return predRating;
			
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
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

				Integer userInnerId = Integer.parseInt(data[0]);

				double[] persDim = { Double.parseDouble(data[5]), Double.parseDouble(data[6]),
						Double.parseDouble(data[7]), Double.parseDouble(data[8]), Double.parseDouble(data[9]) };
				
					userInnerId = rateDao.getUserId(userInnerId.toString());
					persMap.put(userInnerId, persDim);


			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException iex) {
			iex.printStackTrace();
		}

	}

	/**
	 * Build personality based similarity matrix using Euclidean distance measure
	 * 
	 */
	public void buildPersCorrs() {
		persCorrs = new SymmMatrix(persMap.size());

		for (int row = 0; row < persMap.size(); row++) {
			SparseVector iv = new SparseVector(persMap.get(row).length, persMap.get(row));

			for (int col = row + 1; col < persMap.size(); col++) {
				SparseVector jv = new SparseVector(persMap.get(col).length, persMap.get(col));
				
				double persSim = 1/(1+calculateDistance(iv,jv));
				
				if (!Double.isNaN(persSim)) {
					persCorrs.set(row, col, persSim);
				}
			}
		}

	}
	/**
	 * build rating based similarity matrix using Euclidean distance measure
	 */
	public void buildRatCorrs() {
		
		int count = Recommender.rateDao.numUsers();
		ratCorrs = new SymmMatrix(count);
		
		for (int i = 0; i < count; i++) {
			SparseVector iv = trainMatrix.row(i);
			
			if (iv.getCount() == 0)
				continue;
			// user/item itself exclusive
			for (int j = i + 1; j < count; j++) {
				SparseVector jv = trainMatrix.row(j);

				double ratSim = 1/(1+calculateDistance(iv,jv));
				
				if (!Double.isNaN(ratSim)) {
					ratCorrs.set(i, j, ratSim);
				}
			}
			
		}
	}

	/**
	 * Combine the distance between two vectors to determine similarity
	 */
		
	public double calculateDistance(SparseVector iv, SparseVector jv) {
		
		List<Double> is = new ArrayList<>();
		List<Double> js = new ArrayList<>();

		for (Integer idx : jv.getIndex()) {
			if (iv.contains(idx)) {
				is.add(iv.get(idx));
				js.add(jv.get(idx));
			}
		}
		
		double dist = 0.0;
		for (int x=0; x < is.size(); x++) {
			dist+= (is.get(x)-js.get(x)) * (is.get(x)-js.get(x));				
		}
		return dist;
		
	}
}
