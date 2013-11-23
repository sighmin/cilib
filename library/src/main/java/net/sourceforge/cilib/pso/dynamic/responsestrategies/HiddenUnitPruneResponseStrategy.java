/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic.responsestrategies;

import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.nn.architecture.builder.LayerConfiguration;
import net.sourceforge.cilib.problem.nn.NNDataTrainingProblem;
import net.sourceforge.cilib.pso.dynamic.DynamicParticle;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.Bounds;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.container.TypeList;
import net.sourceforge.cilib.type.types.Blackboard;
import net.sourceforge.cilib.math.StatsTables;

import java.util.List;
import java.util.LinkedList;

/**
 * This response strategy removes hidden units from a particle, and updates it's
 * solution vector bitmask to reflect the currently relevant dimensions/weights.
 *
 * @param <E> some {@link PopulationBasedAlgorithm population based algorithm}
 */
public class HiddenUnitPruneResponseStrategy<E extends SinglePopulationBasedAlgorithm> extends EnvironmentChangeResponseStrategy {

    // statics
    public static double significance = 0.01;

    public HiddenUnitPruneResponseStrategy() {

    }

    public HiddenUnitPruneResponseStrategy(HiddenUnitPruneResponseStrategy<E> rhs) {
        super(rhs);
    }

    @Override
    public HiddenUnitPruneResponseStrategy<E> getClone() {
        return new HiddenUnitPruneResponseStrategy<E>(this);
    }


    /**
     * Adds a single neuron to a new hidden layer in the cascade network and adds
     * the new dimensions to the particles, initialised as Double.NaN.
     *
     * {@inheritDoc}
     */
    @Override
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void performReaction(A algorithm, Entity entity) {

        // get required variables
        HeterogeneousNNChargedParticle particle = (HeterogeneousNNChargedParticle) entity;
        NNDataTrainingProblem problem = (NNDataTrainingProblem) algorithm.getOptimisationProblem();
        NeuralNetwork network = problem.getNeuralNetwork();
        fj.data.List<? extends Entity> particles = algorithm.getTopology();
        int I = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(0).getSize(); // input layer size
        int J = getSizeOfLargestHiddenLayerInPopulation(particles);
        int K = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(2).getSize(); // output layer size

        // get required particle variables
        Double VARIANCE_OUTPUT = particle.getVARIANCE_OUTPUT(); // Fix: or must it always begin as 0.001?
        List<Double> variances = particle.getVariances();
        List<Integer> hiddenPositions = particle.getHiddenPositions();
        List<Integer> hiddenIndexes   = particle.getHiddenIndexes();
        int degreesOfFreedom = problem.getValidationSet().size() - 1;

        // while nothing is pruned or sensitivities are high enough (all hidden units are declared 'relevant')
        boolean increase = true;
        while (increase){
            // calculate gammas
            LinkedList<Double> gammas = calcGammas((LinkedList<Double>)variances, VARIANCE_OUTPUT, degreesOfFreedom);

            // arrange gamma values in increasing order
            sortMultipleArrays(variances, hiddenPositions, hiddenIndexes);

            // find gamma_critical from Chi-Squared
            Double gammaCritical = StatsTables.chisqrDistribution(degreesOfFreedom, 1 - significance); // Fix: check with Filipe about alpha param

            // for each gamma value
            for (int i = 0; i < variances.size(); ++i){
                if (Double.isNaN(variances.get(i))) { continue; } // no hidden unit exists at this index
                // if gamma(j) < gamma_critical
                if (variances.get(i) < gammaCritical){
                    pruneParticle(
                        particle, 
                        hiddenIndexes.get(i),
                        hiddenPositions.get(i), 
                        I, J, K);
                    // remove hidden unit information
                    variances.remove(i);
                    hiddenPositions.remove(i);
                    hiddenIndexes.remove(i);
                }
            }
            
            // increase sigma_o if no pruning happened
            for (int i = 0; i < variances.size(); ++i){
                if (Double.isNaN(variances.get(i))) { continue; } // no hidden unit exists at this index
                if (variances.get(i) <= gammaCritical){
                    increase = false;
                    break;
                }
            }
            // set variance output change on particle - Fix: may not be necessary
            if (increase){
                VARIANCE_OUTPUT *= 10;
                particle.setVARIANCE_OUTPUT(VARIANCE_OUTPUT);
            }
        }
    }

    private LinkedList<Double> calcGammas(LinkedList<Double> variances, double VARIANCE_OUTPUT, int degreesOfFreedom){
        LinkedList<Double> gammas = new LinkedList<Double>();
        for (Double d : variances){
            double gamma = (degreesOfFreedom * d) / VARIANCE_OUTPUT;
            gammas.add(gamma);
        }

        return gammas;
    }

    private void pruneParticle(Particle particle, int index, int pos, int I, int J, int K){
        // return if no hidden units left to prune
        if (((HeterogeneousNNChargedParticle)particle).getNumHiddenUnits().intValue() < 2){
            return;
        }

        /* Logically prune particle */
        // prune each vector of the particle & update bitmask
        Bounds bounds = ((Vector) particle.getCandidateSolution()).get(0).getBounds();

        // initial value for what is considered irrelevant dimensions in the vector wrt this particles architecture
        Real init_val          = Real.valueOf(0.0, bounds);
        Int irrelevant_boolean = Int.valueOf(0);

        // get vectors
        Vector candidateSolution = (Vector) particle.getProperties().get(EntityType.CANDIDATE_SOLUTION);
        Vector previousSolution  = (Vector) particle.getProperties().get(EntityType.PREVIOUS_SOLUTION);
        Vector bestPosition      = (Vector) particle.getProperties().get(EntityType.Particle.BEST_POSITION);
        Vector velocity          = (Vector) particle.getProperties().get(EntityType.Particle.VELOCITY);
        Vector bitmask           = (Vector) particle.getProperties().get(EntityType.HeteroNN.BITMASK);

        // update particle vectors with new vectors
        particle.getProperties().put(EntityType.CANDIDATE_SOLUTION,     pruneVector(candidateSolution, index, pos, init_val, I, J, K));
        particle.getProperties().put(EntityType.PREVIOUS_SOLUTION,      pruneVector(previousSolution, index, pos, init_val, I, J, K));
        particle.getProperties().put(EntityType.Particle.BEST_POSITION, pruneVector(bestPosition, index, pos, init_val, I, J, K));
        particle.getProperties().put(EntityType.Particle.VELOCITY,      pruneVector(velocity, index, pos, init_val, I, J, K));
        particle.getProperties().put(EntityType.HeteroNN.BITMASK,       pruneVector(bitmask, index, pos, irrelevant_boolean, I, J, K));

        /* Physically prune particle */
        int num_hidden = ((Int) particle.getProperties().get(EntityType.HeteroNN.NUM_HIDDEN)).intValue();
        particle.getProperties().put(EntityType.HeteroNN.NUM_HIDDEN, Int.valueOf(num_hidden - 1));
        // update 
    }

    private Vector pruneVector(Vector v, int index, int pos, Numeric init_val, int I, int J, int K){
        // jump to hidden vector position
        int limit = pos * (I+1);
        int i = 0;
        while (i < limit){
            ++i;
        }

        // remove hidden unit weights
        limit += I+1;
        while (i < limit){
            v.set(i, init_val);
            ++i;
        }

        // remove hidden-output weights
        // jump to base <- J*(I+1)
        int base = J * (I+1);
        i = base;

        // for k = 0..K
        int k = 0;
        while (k < K){
            v.set(base + (k * (J+1)) + pos, init_val); // To do: replace with non-deprecated method
            ++k;
        }

        return v;
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

    private void sortMultipleArrays(List<Double> masterList, List<Integer>... slaveLists){/*<T extends Comparable<T>>*/
        // bubble sort since the lists are short (size = num hidden units)
        int n = masterList.size();
        for (int c = 0; c < ( n - 1 ); c++) {
            for (int d = 0; d < n - c - 1; d++) {
                if (masterList.get(d).compareTo(masterList.get(d+1)) > 0){ /* For descending order use < */
                    // swap master
                    swap(masterList, d, d+1);

                    // swap slaves
                    for (List<Integer> list : slaveLists){
                        swap(list, d, d+1); //Comparable
                    }
                }
            }
        }
    }

    private <T> void swap(List<T> array, int i1, int i2){
        T temp = array.get(i1);
        array.set(i1, array.get(i2));
        array.set(i2, temp);
    }

    @Override
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void performReaction(A algorithm) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
