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
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.Topologies;
import net.sourceforge.cilib.entity.comparator.SocialBestFitnessComparator;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;

import com.google.common.collect.Lists;

/**
 * Calculates the MSE training error of the best solution of an {@link Algorithm}
 * optimising a {@link NNTrainingProblem}.
 */
public class HeterogeneousNNArchitectureMSETrainingError implements Measurement {

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
        // get problem, data and neural network objects
        NNTrainingProblem problem = (NNTrainingProblem) algorithm.getOptimisationProblem();
        StandardPatternDataTable dataset = problem.getTrainingSet();
        NeuralNetwork neuralNetwork = problem.getNeuralNetwork();
        HeterogeneousNNChargedParticle bestParticle = (HeterogeneousNNChargedParticle)
            Topologies.getBestEntity( ((SinglePopulationBasedAlgorithm)algorithm).getTopology(), new SocialBestFitnessComparator<Particle>() );

        // rebuild NN architecture if number of hidden units differs
        Numeric numHidden = bestParticle.getBestNumHiddenUnits();
        Numeric currentNumHidden = Int.valueOf(neuralNetwork.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(1).getSize());
        if (numHidden.compareTo(currentNumHidden) != 0){
            neuralNetwork.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(1).setSize(numHidden.intValue());
            neuralNetwork.initialise();
        }

        // set weight vector from best particle
        neuralNetwork.setWeights(bestParticle.getBestSolutionWeightVector());

        double mse = 0.0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : dataset) {
            neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            for (Numeric real : error) {
                mse += real.doubleValue() * real.doubleValue();
            }
        }
        mse /= dataset.getNumRows() * error.size();
        return Real.valueOf(mse);
    }
}