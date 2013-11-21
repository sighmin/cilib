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
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.Bounds;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Randomisable;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;

/**
 * This response strategy grows the hidden layer of a particle with a heterogenous
 * NN architecture.
 *
 * @param <E> some {@link PopulationBasedAlgorithm population based algorithm}
 */
public class HiddenLayerGrowResponseStrategy<E extends SinglePopulationBasedAlgorithm> extends EnvironmentChangeResponseStrategy {
    public HiddenLayerGrowResponseStrategy() {
    }

    public HiddenLayerGrowResponseStrategy(HiddenLayerGrowResponseStrategy<E> rhs) {
        super(rhs);
    }

    @Override
    public HiddenLayerGrowResponseStrategy<E> getClone() {
        return new HiddenLayerGrowResponseStrategy<E>(this);
    }

    /**
     * Adds a single unit to a new hidden layer in the particles NN architecture.
     * Adds the appropriate dimensions to the particle & updates the swarms particle
     * sizes, and bitmasks to reflect relevant dimensions for each individual particle.
     *
     * {@inheritDoc}
     */
    @Override
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void performReaction(A algorithm, Entity entity) {

        HeterogeneousNNChargedParticle targetParticle = (HeterogeneousNNChargedParticle) entity;
        NNDataTrainingProblem problem = (NNDataTrainingProblem) algorithm.getOptimisationProblem();
        NeuralNetwork network = problem.getNeuralNetwork();
        fj.data.List<? extends Entity> particles = algorithm.getTopology();
        int I = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(0).getSize(); // input layer size
        int J = getSizeOfLargestHiddenLayerInPopulation(particles);
        int K = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(2).getSize(); // output layer size

        if (targetParticle.isSolutionVectorSaturated()){
            // grow all other particles logically(size)
            for (Entity p : particles){
                HeterogeneousNNChargedParticle currentParticle = (HeterogeneousNNChargedParticle) p;
                if (currentParticle != targetParticle){ // don't logically grow the target particle yet
                    growParticleLogically(currentParticle, I, J, K);
                }
            }

            // grow entity logically(size) & physically(hidden units)
            growParticlePhysically(targetParticle, I, J, K);

        } else {
            // find first open positions for new weights
            growUnsaturatedParticle(targetParticle, I, J, K);
        }
    }

    private void growUnsaturatedParticle(Particle particle, int I, int J, int K){
        /* Logically grow particle */
        Bounds bounds = ((Vector) particle.getCandidateSolution()).get(0).getBounds();

        // initial value for what is considered relevant dimensions in the vector wrt this particles architecture
        Real init_val        = Real.valueOf(Double.NaN, bounds);
        Int relevant_boolean = Int.valueOf(1);

        // get vectors
        Vector candidateSolution = (Vector) particle.getProperties().get(EntityType.CANDIDATE_SOLUTION);
        Vector previousSolution  = (Vector) particle.getProperties().get(EntityType.PREVIOUS_SOLUTION);
        Vector bestPosition      = (Vector) particle.getProperties().get(EntityType.Particle.BEST_POSITION);
        Vector velocity          = (Vector) particle.getProperties().get(EntityType.Particle.VELOCITY);
        Vector bitmask           = (Vector) particle.getProperties().get(EntityType.HeteroNN.BITMASK);

        // update particle vectors with new vectors
        particle.getProperties().put(EntityType.CANDIDATE_SOLUTION,     initNaNElements( growUnsaturated(candidateSolution, bitmask, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.PREVIOUS_SOLUTION,      initNaNElements( growUnsaturated(candidateSolution, bitmask, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.Particle.BEST_POSITION, initNaNElements( growUnsaturated(candidateSolution, bitmask, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.Particle.VELOCITY,      initNaNElements( growUnsaturated(candidateSolution, bitmask, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.HeteroNN.BITMASK,       growUnsaturated(candidateSolution, bitmask, relevant_boolean, I, J, K));

        /* Physically grow particle */
        int num_hidden = ((Int) particle.getProperties().get(EntityType.HeteroNN.NUM_HIDDEN)).intValue();
        particle.getProperties().put(EntityType.HeteroNN.NUM_HIDDEN, Int.valueOf(num_hidden + 1));
    }

    private Vector growUnsaturated(Vector vector, Vector bitmask, Numeric init_value, int I, int J, int K){

        // assign pos <- first irrelevant slot        
        int index = 0;
        int pos = 0;
        while (index < vector.size()){
            if (bitmask.get(index).intValue() == 0){ // 0 indicates an irrelevant corresponding weight in the vector
                break;
            }
            ++pos;
            ++index;
        }

        // find h, the index number of the first open slot's hidden unit
        int h = (int) Math.floor((double) pos / (I+1));

        // fill vector for (I+1) steps with init value
        int limit = I + 1;
        while (index < limit){
            vector.set(index, init_value); // To do: replace with non-deprecated method
            ++index;
        }

        // jump to base <- J*(I+1)
        int base = J * (I+1);
        index = base;

        // for k = 0..K
        int k = 0;
        while (k < K){
            // insert 1 Double.NaN at base + (k*(J+1)) + h
            vector.set(base + (k * (J+1)) + h, init_value); // To do: replace with non-deprecated method
            ++k;
        }

        return vector;
    }

    private void growParticleLogically(Particle particle, int I, int J, int K){
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
        particle.getProperties().put(EntityType.CANDIDATE_SOLUTION,     grow(candidateSolution, init_val, I, J, K));
        particle.getProperties().put(EntityType.PREVIOUS_SOLUTION,      grow(candidateSolution, init_val, I, J, K));
        particle.getProperties().put(EntityType.Particle.BEST_POSITION, grow(candidateSolution, init_val, I, J, K));
        particle.getProperties().put(EntityType.Particle.VELOCITY,      grow(candidateSolution, init_val, I, J, K));
        particle.getProperties().put(EntityType.HeteroNN.BITMASK,       grow(candidateSolution, irrelevant_boolean, I, J, K));
    }

    private void growParticlePhysically(Particle particle, int I, int J, int K){
        /* Logically grow particle */
        Bounds bounds = ((Vector) particle.getCandidateSolution()).get(0).getBounds();

        // initial value for what is considered relevant dimensions in the vector wrt this particles architecture
        Real init_val        = Real.valueOf(Double.NaN, bounds);
        Int relevant_boolean = Int.valueOf(1);

        // get vectors
        Vector candidateSolution = (Vector) particle.getProperties().get(EntityType.CANDIDATE_SOLUTION);
        Vector previousSolution  = (Vector) particle.getProperties().get(EntityType.PREVIOUS_SOLUTION);
        Vector bestPosition      = (Vector) particle.getProperties().get(EntityType.Particle.BEST_POSITION);
        Vector velocity          = (Vector) particle.getProperties().get(EntityType.Particle.VELOCITY);
        Vector bitmask           = (Vector) particle.getProperties().get(EntityType.HeteroNN.BITMASK);

        // update particle vectors with new vectors
        particle.getProperties().put(EntityType.CANDIDATE_SOLUTION,     initNaNElements( grow(candidateSolution, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.PREVIOUS_SOLUTION,      initNaNElements( grow(candidateSolution, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.Particle.BEST_POSITION, initNaNElements( grow(candidateSolution, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.Particle.VELOCITY,      initNaNElements( grow(candidateSolution, init_val, I, J, K)) );
        particle.getProperties().put(EntityType.HeteroNN.BITMASK,       grow(candidateSolution, relevant_boolean, I, J, K));

        /* Physically grow particle */
        int num_hidden = ((Int) particle.getProperties().get(EntityType.HeteroNN.NUM_HIDDEN)).intValue();
        particle.getProperties().put(EntityType.HeteroNN.NUM_HIDDEN, Int.valueOf(num_hidden + 1));
    }

    private Vector grow(Vector current_vector, Numeric init_value, int I, int J, int K){
        Vector.Builder new_vector = Vector.newBuilder();

        int index = 0;
        int old_index = 0;
        // add current hidden units weights
        int limit = J * (I+1);
        while (index < limit){
            new_vector.add(current_vector.get(index));
            ++index;
            ++old_index;
        }
        // add weights of new hidden unit
        limit += I+1;
        while (index < limit){
            new_vector.add(init_value);
            ++index;
        }

        // add weights between hidden unit and output layer
        int k = 0;
        while (k < K){
            int j = 0;
            while (j < J){
                new_vector.add(current_vector.get(old_index));
                ++old_index;
            }
            new_vector.add(init_value);
            new_vector.add(old_index);
            ++k;
        }

        return new_vector.build();
    }

    private Vector initNaNElements(Vector vector){
        for (int i = 0; i < vector.size(); ++i) {
            // if (Double.isNaN(vector.doubleValueOf(i)) { // assume they're all instances of randomisable type
            if (Double.isNaN(vector.doubleValueOf(i)) && vector.get(i) instanceof Randomisable) {
                vector.get(i).randomise();
            }
        }
        return vector;
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

    @Override
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void performReaction(A algorithm) {
        throw new UnsupportedOperationException("Not implemented.");
    }
}
