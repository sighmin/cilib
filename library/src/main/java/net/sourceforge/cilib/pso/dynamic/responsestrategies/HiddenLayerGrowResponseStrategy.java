/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic.responsestrategies;

import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.nn.architecture.builder.LayerConfiguration;
import net.sourceforge.cilib.problem.nn.NNDataTrainingProblem;
import net.sourceforge.cilib.pso.dynamic.DynamicParticle;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.Bounds;
import net.sourceforge.cilib.type.types.Real;

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


        // grow!
        
        












        // NNDataTrainingProblem problem = (NNDataTrainingProblem)algorithm.getOptimisationProblem();
        // NeuralNetwork network = problem.getNeuralNetwork();

        // //add one new neuron to a new hidden layer
        // LayerConfiguration targetLayerConfiguration = new LayerConfiguration();
        // targetLayerConfiguration.setSize(1);
        // network.getArchitecture().getArchitectureBuilder().addLayer(network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().size()-1, targetLayerConfiguration);
        // network.initialise();

        // //add new weights to all the particles in a manner that preserves the old weights
        // fj.data.List<? extends Entity> particles = algorithm.getTopology();
        // int nrOfLayers = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().size();
        // int hiddenLayerSize = nrOfLayers -2;
        // int outputLayerSize = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(nrOfLayers-1).getSize();
        // int inputLayerSize = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(0).getSize();
        // if (network.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(0).isBias()) {
        //     inputLayerSize += 1;
        // }
        // for (Entity curParticle : particles) {
        //     DynamicParticle curDynamicParticle = (DynamicParticle)curParticle;
        //     Bounds bounds = ((Vector) curDynamicParticle.getCandidateSolution()).get(0).getBounds();

        //     //add weights of new neuron
        //     int addPosition = inputLayerSize * (hiddenLayerSize-1);
        //     addPosition += (hiddenLayerSize-2)*(hiddenLayerSize-1)/2;
        //     for (int i = 0; i < inputLayerSize + hiddenLayerSize-1; ++i) {
        //         ((Vector) curDynamicParticle.getCandidateSolution()).insert(addPosition, Real.valueOf(Double.NaN, bounds));
        //         curDynamicParticle.getBestPosition().insert(addPosition, Real.valueOf(Double.NaN, bounds));
        //         curDynamicParticle.getVelocity().insert(addPosition, Real.valueOf(Double.NaN, bounds));
        //     }

        //     //add weights between new neuron and output layer
        //     int startAddPosition = inputLayerSize * hiddenLayerSize;
        //     startAddPosition += (hiddenLayerSize-1)*hiddenLayerSize/2;
        //     for (int curOutput = 0; curOutput < outputLayerSize; ++curOutput) {
        //         addPosition = startAddPosition + (curOutput+1)*(inputLayerSize + hiddenLayerSize) -1;

        //         ((Vector) curDynamicParticle.getCandidateSolution()).insert(addPosition, Real.valueOf(Double.NaN, bounds));
        //         curDynamicParticle.getBestPosition().insert(addPosition, Real.valueOf(Double.NaN, bounds));
        //         curDynamicParticle.getVelocity().insert(addPosition, Real.valueOf(Double.NaN, bounds));
        //     }
        // }
    }

    @Override
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void performReaction(A algorithm) {
    }
}
