/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic;

import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.problem.Problem;
import net.sourceforge.cilib.problem.nn.NNTrainingProblem;
import net.sourceforge.cilib.problem.nn.EntitySpecificNNSlidingWindowTrainingProblem;
import net.sourceforge.cilib.problem.Problem;
import net.sourceforge.cilib.problem.solution.InferiorFitness;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.math.Stats;


import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;

/**
 * Charged Particle used by charged PSO (ChargedVelocityProvider). The only
 * difference from DynamicParticle is that a charged particle stores the charge
 * magnitude and the initialisation strategy for charge.
 *
 * HeterogeneousNNChargedParticle contains a different NN architecture per entity,
 * as well as stores conditions on which the algorithm may grow/prune the size of
 * the architecture.
 *
 * The default error measure is % incorrectly classified (minimization problem),
 * for classification problems.
 */
public class HeterogeneousNNChargedParticle extends ChargedParticle {

    protected static int trendLength            = 20;
    protected static int robelsTrendLength      = 20;
    protected static int overfittingTrendLength = 20;
    // following are used for grow detection 
    protected List<Double> errorTrendTraining;
    protected List<Double> errorTrendValidation;
    // following are used for overfitting detection
    protected List<Double> robelsFactorList;
    protected List<Double> validationErrorList;
    protected double previousPhi;
    // sensitivity analysis variables
    protected Double VARIANCE_OUTPUT = 0.0;
    protected List<Double>  variances       = new ArrayList<Double>();
    protected List<Integer> hiddenPositions = new ArrayList<Integer>(); // logical identifiers for Yj - it's position in the array that's possibly larger than it's architecture
    protected List<Integer> hiddenIndexes   = new ArrayList<Integer>();   // physical identifiers for Yj - means actual j of the architecture

    public HeterogeneousNNChargedParticle() {
        this.errorTrendTraining = new LinkedList<Double>();
        this.errorTrendValidation = new LinkedList<Double>();
        this.robelsFactorList = new LinkedList<Double>();
        this.validationErrorList = new LinkedList<Double>();
    }

    public HeterogeneousNNChargedParticle(HeterogeneousNNChargedParticle copy) {
        super(copy);
        // copy trends
        this.errorTrendTraining = new LinkedList<Double>();
        this.errorTrendValidation = new LinkedList<Double>();
        this.robelsFactorList = new LinkedList<Double>();
        this.validationErrorList = new LinkedList<Double>();
        for (int i = 0; i < errorTrendTraining.size(); ++i){
            this.errorTrendTraining.add(copy.errorTrendTraining.get(i));
            this.errorTrendValidation.add(copy.errorTrendValidation.get(i));
            this.robelsFactorList.add(copy.robelsFactorList.get(i));
            this.validationErrorList.add(copy.validationErrorList.get(i));
        }
    }

    @Override
    public HeterogeneousNNChargedParticle getClone() {
        return new HeterogeneousNNChargedParticle(this);
    }

    @Override
    public void initialise(Problem problem) {
        super.initialise(problem);

        // hidden units
        int num_hidden = ((NNTrainingProblem) problem).getNeuralNetwork().
            getArchitecture().
            getArchitectureBuilder().
            getLayerConfigurations().
            get(1).getSize();
        this.getProperties().put(EntityType.HeteroNN.NUM_HIDDEN, Int.valueOf(num_hidden));
        this.getProperties().put(EntityType.HeteroNN.BITMASK, Vector.fill(Int.valueOf(1), getCandidateSolution().size()) );
        this.getProperties().put(EntityType.HeteroNN.BEST_BITMASK, Vector.copyOf(this.getBitMask()) );
    }

    /**
     * Determines whether the solution vector has irrelevant dimension that can be used to 
     * grow the network, determined using the bitmask 1 as relevant value, and 0 as irrelevant.
     * @return boolean if the solution vector is saturated with relevant values or not
     */
    public boolean isSolutionVectorSaturated(){
        Vector bitmask = (Vector) this.getProperties().get(EntityType.HeteroNN.BITMASK);
        for (int i = 0; i < bitmask.size(); ++i){
            if (bitmask.get(i).intValue() == 0){
                return false;
            }
        }
        return true;
    }

    public void setNumHiddenUnits(int num_hidden){
        this.getProperties().put(EntityType.HeteroNN.NUM_HIDDEN, Int.valueOf(num_hidden));
    }

    /**
     * @return the number of hidden units of this particle's architecture
     */
    public Int getNumHiddenUnits(){
        return (Int) this.getProperties().get(EntityType.HeteroNN.NUM_HIDDEN);
    }

    /**
     * @return the bitmask indicating relevant dimensions of the solution vector
     */
    public Vector getBitMask(){
        return (Vector) this.getProperties().get(EntityType.HeteroNN.BITMASK);
    }

    /**
     * @return the bitmask indicating relevant dimensions of the solution vector
     */
    public Vector getBestBitMask(){
        return (Vector) this.getProperties().get(EntityType.HeteroNN.BEST_BITMASK);
    }

    /**
     * Builds a weight vector with only this particles architectures relevant weights
     * @return the built weight vector
     */
    public Vector getWeightVector(){
        Vector maskedSolution = (Vector) this.getCandidateSolution();
        Vector mask = (Vector) this.getBitMask();
        Vector.Builder weightVectorBuilder = Vector.newBuilder();

        for (int i = 0; i < maskedSolution.size(); ++i){
            int bit = mask.get(i).intValue();
            if (bit == 1){
                weightVectorBuilder.add(maskedSolution.get(i));
            }
        }
        return weightVectorBuilder.build();
    }

    /**
     * Builds a weight vector with only this particles architectures relevant weights
     * for the best position, using the best bitmask, of this particle
     * @return the built weight vector
     */
    public Vector getBestSolutionWeightVector(){
        Vector maskedSolution = (Vector) this.getProperties().get(EntityType.Particle.BEST_POSITION);
        Vector mask = (Vector) this.getBestBitMask();
        Vector.Builder weightVectorBuilder = Vector.newBuilder();

        for (int i = 0; i < maskedSolution.size(); ++i){
            int bit = mask.get(i).intValue();
            if (bit == 1){
                weightVectorBuilder.add(maskedSolution.get(i));
            }
        }
        return weightVectorBuilder.build();
    }
    /**
     * Updates the particle specific error trends over the trend length for all
     * partitioned data sets training, generalisation and validation.
     */
    public void updateErrorTrends(Problem problem){
        EntitySpecificNNSlidingWindowTrainingProblem nnProblem = (EntitySpecificNNSlidingWindowTrainingProblem) problem;

        // collect error measures
        double trainingError = getFitness().getValue(); // fitness is calculated on an entity basis, as an MSE measure
        double validationError = nnProblem.getMSEValidationError(this);
        double robels_factor = validationError / trainingError;
        updatePreviousPhi();

        // update error trends
        int length = errorTrendTraining.size();
        if (length < trendLength){
            errorTrendTraining.add(Double.valueOf(trainingError));
            errorTrendValidation.add(Double.valueOf(validationError));
            robelsFactorList.add(Double.valueOf(robels_factor));
        } else {
            ((LinkedList) errorTrendTraining).remove(); // removes head of list
            errorTrendTraining.add(Double.valueOf(trainingError)); // adds to tail of list
            ((LinkedList) errorTrendValidation).remove();
            errorTrendValidation.add(Double.valueOf(validationError));
            ((LinkedList) robelsFactorList).remove();
            robelsFactorList.add(Double.valueOf(robels_factor));
        }

        // update validationErrorList for overfitting detection
        length = validationErrorList.size();
        if (length < overfittingTrendLength){
            validationErrorList.add(Double.valueOf(validationError));
        } else {
            ((LinkedList) validationErrorList).remove();
            validationErrorList.add(Double.valueOf(validationError));
        }

        // update robels factor list for overfitting detection
        length = robelsFactorList.size();
        if (length < robelsTrendLength){
            robelsFactorList.add(Double.valueOf(robels_factor));
        } else {
            ((LinkedList) robelsFactorList).remove();
            robelsFactorList.add(Double.valueOf(robels_factor));
        }
    }

    private void updatePreviousPhi(){
        double previousPreviousPhi = previousPhi;
        double avg_plus_stdev = getAverageMovingRobelsFactor() + getSTDEVRobelsFactor();
        previousPhi = Math.min(Math.min(previousPreviousPhi, avg_plus_stdev), 1.0); // min = {prev, avg+stdev, 1.0}
    }

    public Double getAverageMovingOverfittingValidationError(){
        return Stats.mean(validationErrorList);
    }

    public Double getAverageMovingRobelsFactor(){
        return Stats.mean(robelsFactorList);
    }

    public Double getAverageMovingTrainingError(){
        return Stats.mean(errorTrendTraining);
    }

    public Double getAverageMovingValidationError(){
        return Stats.mean(errorTrendValidation);
    }

    public Double getSTDEVOverfittingValidationError(){
        return Stats.stdDev(validationErrorList);
    }

    public Double getSTDEVRobelsFactor(){
        return Stats.stdDev(robelsFactorList);
    }

    public Double getSTDEVMovingTrainingError(){
        return Stats.stdDev(errorTrendTraining);
    }

    public Double getSTDEVMovingValidationError(){
        return Stats.stdDev(errorTrendValidation);
    }

    public void setTrendLength(int trendLength){
        this.trendLength = trendLength;
    }

    public int getTrendLength(){
        return this.trendLength;
    }

    public void setRobelsTrendLength(int robelsTrendLength){
        this.robelsTrendLength = robelsTrendLength;
    }

    public int getRobelsTrendLength(){
        return this.robelsTrendLength;
    }

    public void setOverfittingTrendLength(int overfittingTrendLength){
        this.overfittingTrendLength = overfittingTrendLength;
    }

    public int geOverfittingTrendLength(){
        return this.overfittingTrendLength;
    }

    public List<Double> getErrorTrendTraining(){
        return this.errorTrendTraining;
    }

    public List<Double> getErrorTrendValidation(){
        return this.errorTrendValidation;
    }

    public List<Double> getRobelsFactorList(){
        return this.robelsFactorList;
    }

    public List<Double> getValidationErrorList(){
        return this.validationErrorList;
    }

    public double getPreviousPhi(){
        return this.previousPhi;
    }

    public void setVARIANCE_OUTPUT(Double d){
        this.VARIANCE_OUTPUT = d;
    }

    public Double getVARIANCE_OUTPUT(){
        return this.VARIANCE_OUTPUT;
    }

    public void setVariances(List<Double> variances){
        this.variances = variances;
    }

    public List<Double> getVariances(){
        return this.variances;
    }

    public void setHiddenIndexes(List<Integer> hiddenIndexes){
        this.hiddenIndexes = hiddenIndexes;
    }

    public List<Integer> getHiddenIndexes(){
        return this.hiddenIndexes;
    }

    public void setHiddenPositions(List<Integer> hiddenPositions){
        this.hiddenPositions = hiddenPositions;
    }

    public List<Integer> getHiddenPositions(){
        return this.hiddenPositions;
    }
}
