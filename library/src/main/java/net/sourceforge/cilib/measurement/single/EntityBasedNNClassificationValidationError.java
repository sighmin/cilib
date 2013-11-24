/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.measurement.single;

import net.sourceforge.cilib.algorithm.Algorithm;
import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.measurement.Measurement;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.nn.architecture.visitors.OutputErrorVisitor;
import net.sourceforge.cilib.problem.nn.NNTrainingProblem;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.entity.Topologies;
import net.sourceforge.cilib.entity.comparator.SocialBestFitnessComparator;

/**
 * Calculates the of patterns classified incorrecly for an optimization algorithm
 * optimizing a {@link NNDataTrainingProblem}. The output range is in [0.0, 1.0].
 */
public class EntityBasedNNClassificationValidationError implements Measurement {

    protected ControlParameter outputSensitivityThreshold = ConstantControlParameter.of(0.2);

    /**
     * {@inheritDoc }
     */
    @Override
    public Measurement getClone() {
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Type getValue(Algorithm algorithm) {
        NNTrainingProblem problem = (NNTrainingProblem) algorithm.getOptimisationProblem();
        StandardPatternDataTable dataSet = problem.getValidationSet();
        NeuralNetwork neuralNetwork = problem.getNeuralNetwork();
        HeterogeneousNNChargedParticle particle = (HeterogeneousNNChargedParticle)
            Topologies.getBestEntity( ((SinglePopulationBasedAlgorithm)algorithm).getTopology(), new SocialBestFitnessComparator<Particle>() );

        // rebuild NN architecture if number of hidden units differs
        Numeric numHidden = particle.getBestNumHiddenUnits();
        Numeric currentNumHidden = Int.valueOf(neuralNetwork.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(1).getSize());
        if (numHidden.compareTo(currentNumHidden) != 0){
            neuralNetwork.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(1).setSize(numHidden.intValue());
            neuralNetwork.initialise();
        }
        // set weight vector from best particle
        neuralNetwork.setWeights(particle.getBestSolutionWeightVector());

        int numberPatternsCorrect = 0;
        int numberPatternsIncorrect = 0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : dataSet) {
            neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            boolean isCorrect = true;
            
            for (Numeric real : error) {
                if (Math.abs(real.doubleValue()) > this.outputSensitivityThreshold.getParameter()) {
                    isCorrect = false;
                    break;
                }
            }
            if (isCorrect){
                numberPatternsCorrect++;
            } else {
                numberPatternsIncorrect++;
            }
        }

        double percentageIncorrect = (double) numberPatternsIncorrect / ((double) numberPatternsIncorrect + (double) numberPatternsCorrect);
        return Real.valueOf(percentageIncorrect);
    }

    /**
     * Method to set the output sensitivity threshold value for determining what
     * NN output real value constitutes a correct/incorrect classification
     * @param outputSensitivityThreshold The threshold for NN output classification
     */
    public void setOuputSensitivityThreshold(double outputSensitivityThreshold){
        this.outputSensitivityThreshold = ConstantControlParameter.of(outputSensitivityThreshold);
    }
}
