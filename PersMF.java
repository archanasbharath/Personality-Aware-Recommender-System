package librec.personality;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import happy.coding.io.FileIO;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
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
* NOTE: To have more control on learning, you can add additional regularation parameters to user/item biases. For
* simplicity, we do not do this.
* 
* @author guoguibing
* 
*/
public class PersMF extends IterativeRecommender {

	private SparseMatrix binaryMatrix;
	 
	protected DenseVector persD11, persD12, persD13, persD14, persD15, persD16, persD17,
	                      persD21, persD22, persD23, persD24, persD25, persD26, persD27,
	                      persD31, persD32, persD33, persD34, persD35, persD36, persD37,
	                      persD41, persD42, persD43, persD44, persD45, persD46, persD47,
	                      persD51, persD52, persD53, persD54, persD55, persD56, persD57;
	

	private Map<Integer, double[]> persMap;
	
	private DenseVector itemMeans;
	
	
	public PersMF(SparseMatrix rm, SparseMatrix tm, int fold) {
		super(rm, tm, fold);
	}

	protected void initModel() throws Exception {

		super.initModel();
		readData();
		buildBinRatingMatrix();
		
		
		//create personality dimension vectors for all 5 dimensions 
		persD11= new DenseVector(numFactors);
		persD12= new DenseVector(numFactors);
		persD13= new DenseVector(numFactors);
		persD14= new DenseVector(numFactors);
		persD15= new DenseVector(numFactors);
		persD16= new DenseVector(numFactors);
		persD17= new DenseVector(numFactors);
		persD21= new DenseVector(numFactors);
		persD22= new DenseVector(numFactors);
		persD23= new DenseVector(numFactors);
		persD24= new DenseVector(numFactors);
		persD25= new DenseVector(numFactors);
		persD26= new DenseVector(numFactors);
		persD27= new DenseVector(numFactors);
		persD31= new DenseVector(numFactors);
		persD32= new DenseVector(numFactors);
		persD33= new DenseVector(numFactors);
		persD34= new DenseVector(numFactors);
		persD35= new DenseVector(numFactors);
		persD36= new DenseVector(numFactors);
		persD37= new DenseVector(numFactors);
		persD41= new DenseVector(numFactors);
		persD42= new DenseVector(numFactors);
		persD43= new DenseVector(numFactors);
		persD44= new DenseVector(numFactors);
		persD45= new DenseVector(numFactors);
		persD46= new DenseVector(numFactors);
		persD47= new DenseVector(numFactors);
		persD51= new DenseVector(numFactors);
		persD52= new DenseVector(numFactors);
		persD53= new DenseVector(numFactors);
		persD54= new DenseVector(numFactors);
		persD55= new DenseVector(numFactors);
		persD56= new DenseVector(numFactors);
		persD57= new DenseVector(numFactors);

		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		
		// initialize user bias
		userBias.init(initMean, initStd);
		itemBias.init(initMean, initStd);
		
		//initialize the user personality dimensions
		initPersVectors();
		
		itemMeans = new DenseVector(numItems);
		
		for (int j = 0 ; j <numItems; j++) {
			SparseVector h = trainMatrix.column(j);
			itemMeans.set(j,  h.getCount() > 0 ? h.mean():globalMean);
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
				
						
								
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f); 										
										
					double persValuesSum = getPersValues(u,f);
					double delta_u = euj * qjf - regU * puf;
					double delta_j = (euj * puf) + (euj * persValuesSum) - (regI * qjf);

					P.add(u, f, lRate * delta_u );
						
					Q.add(j, f, lRate * delta_j);

					loss += regU * puf * puf + regI * qjf * qjf;
					updatePersVectors(u, euj, qjf,f);	
				}
				
			}
			
			loss *= 0.5;

			
			if (isConverged(iter))
				break;

		}// end of training

	}

	/**
	 * predict the rating of item j by user u 
	 */
	protected double predict(int u, int j) throws Exception {
		
		return itemMeans.get(j) + itemBias.get(j)+userBias.get(u)+dotProd(P, u, Q, j);
	}
	
	
	public double dotProd(DenseMatrix m, int mrow, DenseMatrix n, int nrow) {
		int mcols = m.numColumns();
		int ncols = n.numColumns();
		assert mcols == ncols;	

		double res = 0;
		for (int fac = 0; fac < mcols; fac++) {
			double persPlusUserFactor = 0.0;
			persPlusUserFactor = m.get(mrow,  fac) + getPersValues(mrow, fac);
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
	
	
	/**
	 * Build binary matrix with rating data
	 * 1 - for ratings that are available
	 * 0 - for ratings that are not available
	 */
	
	public void buildBinRatingMatrix() {
	
		binaryMatrix = new SparseMatrix(trainMatrix);
		
		for (MatrixEntry me : binaryMatrix) {
			
			int x = me.row(); // user
			int y = me.column(); // item
			double val1 = 1.0;
			double val0 = 0;
						
			if (me.get() > 0.0) {			
				binaryMatrix.set(x, y, val1);
			}else {
				binaryMatrix.set(x, y, val0);
			}
			
		}
		
	}
	
	/**
	 * Initialize the personality factor vectors with the data
	 * retrieved from the personality input file
	 */
	
	public void initPersVectors() {

		persD11.init(initMean, initStd);
		persD12.init(initMean, initStd);
		persD13.init(initMean, initStd);
		persD14.init(initMean, initStd);
		persD15.init(initMean, initStd);
		persD16.init(initMean, initStd);
		persD17.init(initMean, initStd);
		persD21.init(initMean, initStd);
		persD22.init(initMean, initStd);
		persD23.init(initMean, initStd);
		persD24.init(initMean, initStd);
		persD25.init(initMean, initStd);
		persD26.init(initMean, initStd);
		persD27.init(initMean, initStd);
		persD31.init(initMean, initStd);
		persD32.init(initMean, initStd);
		persD33.init(initMean, initStd);
		persD34.init(initMean, initStd);
		persD35.init(initMean, initStd);
		persD36.init(initMean, initStd);
		persD37.init(initMean, initStd);
		persD41.init(initMean, initStd);
		persD42.init(initMean, initStd);
		persD43.init(initMean, initStd);
		persD44.init(initMean, initStd);
		persD45.init(initMean, initStd);
		persD46.init(initMean, initStd);
		persD47.init(initMean, initStd);
		persD51.init(initMean, initStd);
		persD52.init(initMean, initStd);
		persD53.init(initMean, initStd);
		persD54.init(initMean, initStd);
		persD55.init(initMean, initStd);
		persD56.init(initMean, initStd);
		persD57.init(initMean, initStd);
	}
	
	
	/**
	 * get the sum of all 5 personality factor values for a given user factor
	 * @param a
	 * @param fac
	 * @return
	 */
	public double getPersValues(int a, int fac) {
		
		double persForUser[] = persMap.get(a);
		double persDimSum = 0 ;
		
		for (int d = 0 ; d < persForUser.length; d++) {
			if (Double.compare(persForUser[d],1.0) == 0) {
				
				switch (d){
				case 0:persDimSum += persD11.get(fac);
					break;
				case 1:persDimSum += persD21.get(fac);
					break;
				case 2:persDimSum += persD31.get(fac);
					break;
				case 3:persDimSum += persD41.get(fac);
					break;
				case 4:persDimSum += persD51.get(fac);
					break;
				default:
					break;
					
				}
			}else if (Double.compare(persForUser[d],2.0) == 0) {
				
				switch (d){
				case 0:persDimSum += persD12.get(fac);
					break;
				case 1:persDimSum += persD22.get(fac);
					break;
				case 2:persDimSum += persD32.get(fac);
					break;
				case 3:persDimSum += persD42.get(fac);
					break;
				case 4:persDimSum += persD52.get(fac);
					break;
				default:
					break;
					}						
			}else if (Double.compare(persForUser[d],3.0) == 0) {
				switch (d){
				case 0:
					persDimSum += persD13.get(fac);
					break;
				case 1:
					persDimSum += persD23.get(fac);
					break;
				case 2:
					persDimSum += persD33.get(fac);
					break;
				case 3:
					persDimSum += persD43.get(fac);
					break;
				case 4:
					persDimSum += persD53.get(fac);
					break;
				default:
					break;
					}						
			}else if (Double.compare(persForUser[d],4.0) == 0) {
				switch (d){
				case 0:
					persDimSum += persD14.get(fac);
					break;
				case 1:
					persDimSum += persD24.get(fac);
					break;
				case 2:
					persDimSum += persD34.get(fac);
					break;
				case 3:
					persDimSum += persD44.get(fac);
					break;
				case 4:
					persDimSum += persD54.get(fac);
					break;
				default:
					break;
					}						
			}else if (Double.compare(persForUser[d],5.0) == 0) {
				switch (d){
				case 0:
					persDimSum += persD15.get(fac);
					
					break;
				case 1:
					persDimSum += persD25.get(fac);
					break;
				case 2:
					persDimSum += persD35.get(fac);					
					break;
				case 3:
					persDimSum += persD45.get(fac);
					
					break;
				case 4:
					persDimSum += persD55.get(fac);
					break;
				default:
					break;
					}						
			}else if (Double.compare(persForUser[d],6.0) == 0) {
				switch (d){
				case 0:
					persDimSum += persD16.get(fac);
					break;
				case 1:
					persDimSum += persD26.get(fac);
					break;
				case 2:
					persDimSum += persD36.get(fac);
					break;
				case 3:
					persDimSum += persD46.get(fac);
					break;
				case 4:
					persDimSum += persD56.get(fac);
					break;
				default:
					break;
					}						
			}else if (Double.compare(persForUser[d],7.0) == 0) {
				switch (d){
				case 0:
					persDimSum += persD17.get(fac);
					break;
				case 1:
					persDimSum += persD27.get(fac);
					break;
				case 2:
					persDimSum += persD37.get(fac);
					break;
				case 3:
					persDimSum += persD47.get(fac);
					break;
				case 4:
					persDimSum += persD57.get(fac);
					break;
				default:
					break;
					}						
			}	
						
	}
		return persDimSum;
	}
	
	/**
	 * update personality factor vectors
	 * @param u
	 * @param err
	 * @param qt
	 * @param f
	 */
	public void updatePersVectors(int u, double err, double qt, int f) {
		
		
		double persArray[] = persMap.get(u);
		double persD1 = 0.0;
		double persD2 = 0.0;
		double persD3 = 0.0;
		double persD4 = 0.0;
		double persD5 = 0.0;
		
		for (int d = 0 ; d < persArray.length; d++) {
			if (Double.compare(persArray[d],1.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD11.get(f);
					persD11.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD21.get(f);
					persD21.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD31.get(f);
					persD31.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD41.get(f);
					persD41.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD51.get(f);
					persD51.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
				default:
					break;
					
				}
			}else if (Double.compare(persArray[d],2.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD12.get(f);
					persD12.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD22.get(f);
					persD22.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD32.get(f);
					persD32.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD42.get(f);
					persD42.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD52.get(f);
					persD52.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
				default:
					break;
					
				}					
			}else if (Double.compare(persArray[d],3.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD13.get(f);
					persD13.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD23.get(f);
					persD23.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD33.get(f);
					persD33.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD43.get(f);
					persD43.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD53.get(f);
					persD53.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
				default:
					break;
					
				}				
			}else if (Double.compare(persArray[d],4.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD14.get(f);
					persD14.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD24.get(f);
					persD24.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD34.get(f);
					persD34.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD44.get(f);
					persD44.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD54.get(f);
					persD54.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
					
				}							
			}else if (Double.compare(persArray[d],5.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD15.get(f);
					persD15.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD25.get(f);
					persD25.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD35.get(f);
					persD35.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD45.get(f);
					persD45.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD55.get(f);
					persD55.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
				default:
					break;
					
				}						
			}else if (Double.compare(persArray[d],6.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD16.get(f);
					persD16.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD26.get(f);
					persD26.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD36.get(f);
					persD36.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD46.get(f);
					persD46.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD56.get(f);
					persD56.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
				default:
					break;
					
				}						
			}else if (Double.compare(persArray[d],7.0) == 0) {
				switch (d){
				case 0:
					persD1 = persD17.get(f);
					persD17.add(f, lRate * (err*qt - regU * persD1));
					loss += regU * persD1 * persD1;
					break;
				case 1:
					persD2 = persD27.get(f);
					persD27.add(f, lRate * (err*qt - regU * persD2));
					loss += regU * persD2 * persD2;
					break;
				case 2:
					persD3 = persD37.get(f);
					persD37.add(f, lRate * (err*qt - regU * persD3));
					loss += regU * persD3 * persD3;
					break;
				case 3:
					persD4 = persD47.get(f);
					persD47.add(f, lRate * (err*qt - regU * persD4));
					loss += regU * persD4 * persD4;
					break;
				case 4:
					persD5 = persD57.get(f);
					persD57.add(f, lRate * (err*qt - regU * persD5));
					loss += regU * persD5 * persD5;
					break;
				default:
					break;
					
				}							
			}	
						
	}
		
	}
	
	
}//class ends

