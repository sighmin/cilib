/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.measurement.single;

import net.sourceforge.cilib.algorithm.Algorithm;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.measurement.Measurement;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.nn.architecture.visitors.OutputErrorVisitor;
import net.sourceforge.cilib.problem.nn.NNTrainingProblem;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;

/**
 * Calculates the of patterns classified incorrecly for an optimization algorithm
 * optimizing a {@link NNDataTrainingProblem}. The output range is in [0.0, 1.0].
 */
public class NNClassificationTrainingError implements Measurement {

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
        Vector solution = (Vector) algorithm.getBestSolution().getPosition();
        NNTrainingProblem problem = (NNTrainingProblem) algorithm.getOptimisationProblem();
        StandardPatternDataTable trainingSet = problem.getTrainingSet();
        NeuralNetwork neuralNetwork = problem.getNeuralNetwork();
        neuralNetwork.setWeights(solution);

        int numberPatternsCorrect = 0;
        int numberPatternsIncorrect = 0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : trainingSet) {
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
