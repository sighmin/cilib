/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic.responsestrategies;

import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.topologies.Neighbourhood;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.util.Cloneable;
import net.sourceforge.cilib.entity.Entity;

/**
 * This abstract class defines the interface that particle based response strategies have to adhere to.
 * Detection strategies are used within the scope of a {@link DynamicIterationStrategy} to
 * detect whether the environment has change during the course of an
 * {@link Algorithm algorithm's} execution.
 * 
 * This class exposes a particle based detection method where the environment is
 * specific to the particle.
 */
public abstract class ParticleBasedEnvironmentChangeResponseStrategy implements Cloneable {
    protected boolean hasMemory = true;

    public ParticleBasedEnvironmentChangeResponseStrategy() {
        this.hasMemory = true;
    }

    public ParticleBasedEnvironmentChangeResponseStrategy(ParticleBasedEnvironmentChangeResponseStrategy rhs) {
        this.hasMemory = rhs.hasMemory;
    }

    /**
     * Clone the <tt>ParticleBasedEnvironmentChangeResponseStrategy</tt> object.
     *
     * @return A cloned <tt>ParticleBasedEnvironmentChangeResponseStrategy</tt>
     */
    public abstract ParticleBasedEnvironmentChangeResponseStrategy getClone();

    /**
     * Respond to the environment change and ensure that the neighbourhood best entities are
     * updated. This method (Template Method) calls two other methods in turn:
     * <ul>
     * <li>{@link #performReaction(PopulationBasedAlgorithm)}</li>
     * <li>{@link #updateNeighbourhoodBestEntities(Topology)}</li>
     * </ul>
     *
     * @param algorithm some {@link PopulationBasedAlgorithm population based algorithm}
     */
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void respond(A algorithm, Entity entity) {
        performReaction(algorithm, entity);
        if(hasMemory) {
            updateNeighbourhoodBestEntities(algorithm.getTopology(), algorithm.getNeighbourhood());
        }
    }

    /**
     * This is the method responsible for responding that should be overridden by sub-classes.
     * @param algorithm
     */
    protected abstract <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void performReaction(A algorithm, Entity entity);

    /**
     * TODO: The problem with this is that it is PSO specific. It uses {@link Particle particles}
     * instead of {@link Entity entities}, because the {@link Entity} class does not have the
     * notion of a neighbourhood best.
     *
     * @param topology a topology of {@link Particle particles} :-(
     */
    protected <P extends Particle> void updateNeighbourhoodBestEntities(fj.data.List<P> topology, Neighbourhood<P> neighbourhood) {
        for (P current : topology) {
            current.calculateFitness();
            for (P other : neighbourhood.f(topology, current)) {
                if (current.getSocialFitness().compareTo(other.getNeighbourhoodBest().getSocialFitness()) > 0) {
                    other.setNeighbourhoodBest(current);
                }
            }
        }
    }

    public boolean getHasMemory() {
        return hasMemory;
    }

    public void setHasMemory(boolean hasMemory) {
        this.hasMemory = hasMemory;
    }
}
