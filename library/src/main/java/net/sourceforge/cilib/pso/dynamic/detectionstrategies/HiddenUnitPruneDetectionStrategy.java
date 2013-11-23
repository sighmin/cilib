/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic.detectionstrategies;

import net.sourceforge.cilib.algorithm.Algorithm;
import net.sourceforge.cilib.algorithm.population.HasNeighbourhood;
import net.sourceforge.cilib.algorithm.population.HasTopology;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.nn.architecture.visitors.OutputErrorVisitor;
import net.sourceforge.cilib.problem.nn.NNDataTrainingProblem;
import net.sourceforge.cilib.problem.nn.EntitySpecificNNSlidingWindowTrainingProblem;

import java.util.List;
import java.util.LinkedList;

/**
 * This class implements {@link ParticleBasedEnvironmentChangeDetectionStrategy} 
 * by detecting changes on a per entity basis, that is, the environment is specific
 * to the entity.
 */
public class HiddenUnitPruneDetectionStrategy extends EnvironmentChangeDetectionStrategy {

    private EntitySpecificNNSlidingWindowTrainingProblem problem;
    private StandardPatternDataTable patterns;
    private NeuralNetwork network;
    private int I;
    private int J;
    private int K;

    public HiddenUnitPruneDetectionStrategy() {
    }

    public HiddenUnitPruneDetectionStrategy(EnvironmentChangeDetectionStrategy rhs) {
        super(rhs);
    }

    @Override
    public HiddenUnitPruneDetectionStrategy getClone() {
        return new HiddenUnitPruneDetectionStrategy(this);
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm, Entity entity) {
        if (algorithm.getIterations() != 0 && algorithm.getIterations() % interval == 0) {
            // return if NN is already at it's smallest, and overfitting is a result of something other than too many hidden units
            int num_hidden = ((HeterogeneousNNChargedParticle) entity).getNumHiddenUnits().intValue();
            if (num_hidden < 2){
                return false;
            }

            // collect required objects
            HeterogeneousNNChargedParticle particle = (HeterogeneousNNChargedParticle) entity;
            problem = (EntitySpecificNNSlidingWindowTrainingProblem) algorithm.getOptimisationProblem();
            patterns = problem.getValidationSet();
            problem.reInitArchitecture((Entity) particle);
            network = problem.getNeuralNetwork();
            I = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(0).getSize(); // input layer size
            J = getSizeOfLargestHiddenLayerInPopulation(algorithm.getTopology());
            K = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(2).getSize(); // output layer size

            // prepare hidden unit indexes
            initHiddenUnitTrackingIndexes((Particle)particle);
            List<Double>  variances       = particle.getVariances();
            List<Integer> hiddenPositions = particle.getHiddenPositions();
            List<Integer> hiddenIndexes   = particle.getHiddenIndexes();

            // (c) for each hidden unit Yj
            for (int i = 0; i < variances.size(); ++i){
                if (Double.isNaN(variances.get(i))) { continue; }

                // calc variance
                double variance = getVariance((Particle)particle, hiddenPositions.get(i).intValue());
                ((LinkedList<Double>) variances).set(i, variance);
            }

            // set results on particle blackboard
            particle.setVariances(variances);
            particle.setHiddenPositions(hiddenPositions);
            particle.setHiddenIndexes(hiddenIndexes);

            // (d) apply prune heuristic (part of response strategy)
            return true; // return true regardless, response strategy decides whether to prune based on parameter sensitivity
        }
        return false;
    }

    private double getVariance(Particle particle, int j_pos){
        double sum = 0.0;
        LinkedList<Double> nullities = getNullities(particle, j_pos);
        double nullityAvg = getNullityAverage(nullities);

        for (int p = 0; p < patterns.size(); ++p){
            double difference = nullities.get(p) - nullityAvg;
            sum += Math.pow(difference , 2);
        }

        return sum / patterns.size();
    }

    private double getNullityAverage(LinkedList<Double> nullities){
        double sum = 0.0;
        for (Double d : nullities){
            sum += d;
        }
        return sum / nullities.size();
    }

    private LinkedList<Double> getNullities(Particle particle, int j_pos){
        LinkedList<Double> nullities = new LinkedList<Double>();

        double nullity = 0.0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern p : patterns) {
            network.evaluatePattern(p);
            visitor.setInput(p);
            network.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            int k = 0;
            nullity = 0.0;
            for (Numeric real : error) {
                // real is Ok output for pattern p
                double wkj = getWKJ((Vector) particle.getCandidateSolution(), k, j_pos);
                double sensitivity = real.doubleValue() * (1 - real.doubleValue()) * wkj;
                nullity += sensitivity;
                ++k;
            }
            nullity = nullity / K; // for notational convenience, it's also the size of "error" above
            nullities.add(nullity);
        }
        return nullities;
    }

    private double getWKJ(Vector v, int k, int j_pos){
        double wkj;
        int base = J * (I+1);
        int final_index = base + (k * (J+1)) + j_pos;
        wkj = v.get(final_index).doubleValue();

        return wkj;
    }

    private void initHiddenUnitTrackingIndexes(Particle particle){
        HeterogeneousNNChargedParticle nnParticle = (HeterogeneousNNChargedParticle) particle;
        Vector bitmask = (Vector) particle.getProperties().get(EntityType.HeteroNN.BITMASK);
        nnParticle.getVariances().clear();
        nnParticle.getHiddenPositions().clear();
        nnParticle.getHiddenPositions().clear();

        int i = 0;
        int base = J * (I+1);
        int yj_index = 0;
        int yj_pos = 0;
        while (i < base){
            nnParticle.getHiddenPositions().add(yj_pos);
            if (bitmask.get(i).intValue() == 0){
                nnParticle.getVariances().add(Double.NaN);
                nnParticle.getHiddenIndexes().add(Integer.MIN_VALUE);
            } else {
                nnParticle.getVariances().add(0.0);
                nnParticle.getHiddenIndexes().add(yj_index);
                ++yj_index;
            }

            i += (I+1);
            ++yj_pos;
        }
    }

    private int getSizeOfLargestHiddenLayerInPopulation(fj.data.List<? extends Entity> particles){
        int largest = 0;
        for (Entity p : particles){
            int current = ((HeterogeneousNNChargedParticle) p).getNumHiddenUnits().intValue();
            if (current > largest){
                largest = current;
            }
        }

        return largest;
    }
}
