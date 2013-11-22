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

    protected int trendLength = Integer.MAX_VALUE;
    protected List<Double> errorTrendTraining;
    protected List<Double> errorTrendGeneralisation;
    protected List<Double> errorTrendValidation;
    // sensitivity analysis variables
    protected Double VARIANCE_OUTPUT = 0.0;
    protected List<Double>  variances       = new ArrayList<Double>();
    protected List<Integer> hiddenPositions = new ArrayList<Integer>(); // logical identifiers for Yj - it's position in the array that's possibly larger than it's architecture
    protected List<Integer> hiddenIndexes   = new ArrayList<Integer>();   // physical identifiers for Yj - means actual j of the architecture

    public HeterogeneousNNChargedParticle() {
        this.errorTrendTraining = new LinkedList<Double>();
        this.errorTrendGeneralisation = new LinkedList<Double>();
        this.errorTrendValidation = new LinkedList<Double>();
    }

    public HeterogeneousNNChargedParticle(HeterogeneousNNChargedParticle copy) {
        super(copy);
        this.trendLength = copy.getTrendLength();
        // copy trends
        this.errorTrendTraining = new LinkedList<Double>();
        this.errorTrendGeneralisation = new LinkedList<Double>();
        this.errorTrendValidation = new LinkedList<Double>();
        for (int i = 0; i < errorTrendTraining.size(); ++i){
            this.errorTrendTraining.add(copy.errorTrendTraining.get(i));
            this.errorTrendGeneralisation.add(copy.errorTrendGeneralisation.get(i));
            this.errorTrendValidation.add(copy.errorTrendValidation.get(i));
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
     * Builds a weight vector with only this particles architectures relevant weights
     * @return the built weight vector
     */
    public Vector getWeightVector(){
        // create architecture specific weight vector
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
     * Updates the particle specific error trends over the trend length for all
     * partitioned data sets training, generalisation and validation.
     */
    public void updateErrorTrends(Problem problem){
        EntitySpecificNNSlidingWindowTrainingProblem nnProblem = (EntitySpecificNNSlidingWindowTrainingProblem) problem;

        // collect error measures
        double trainingError = getFitness().getValue(); // fitness is calculated on an entity basis, as an MSE measure
        double generalisationError = nnProblem.getMSEGeneralisationError(this);
        double validationError = nnProblem.getMSEValidationError(this);

        // update error trends
        int length = errorTrendTraining.size();
        if (length < trendLength){
            errorTrendTraining.add(Double.valueOf(trainingError));
            errorTrendGeneralisation.add(Double.valueOf(generalisationError));
            errorTrendValidation.add(Double.valueOf(validationError));
        } else {
            ((LinkedList) errorTrendTraining).remove(); // removes head of list
            errorTrendTraining.add(Double.valueOf(trainingError)); // adds to tail of list
            ((LinkedList) errorTrendGeneralisation).remove(); // removes head of list
            errorTrendGeneralisation.add(Double.valueOf(generalisationError)); // adds to tail of list
            ((LinkedList) errorTrendValidation).remove(); // removes head of list
            errorTrendValidation.add(Double.valueOf(validationError)); // adds to tail of list
        }
    }

    public Double getAverageMovingTrainingError(){
        return Stats.mean(errorTrendTraining);
    }

    public Double getAverageMovingGeneralisationError(){
        return Stats.mean(errorTrendGeneralisation);
    }

    public Double getAverageMovingValidationError(){
        return Stats.mean(errorTrendValidation);
    }

    public Double getSTDEVMovingTrainingError(){
        return Stats.stdDev(errorTrendTraining);
    }

    public Double getSTDEVMovingGeneralisationError(){
        return Stats.stdDev(errorTrendGeneralisation);
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

    public List<Double> getErrorTrendTraining(){
        return this.errorTrendTraining;
    }

    public List<Double> getErrorTrendGeneralisation(){
        return this.errorTrendGeneralisation;
    }

    public List<Double> getErrorTrendValidation(){
        return this.errorTrendValidation;
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
