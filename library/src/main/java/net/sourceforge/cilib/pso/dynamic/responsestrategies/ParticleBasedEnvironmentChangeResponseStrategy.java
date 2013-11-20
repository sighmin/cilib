/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic.responsestrategies;

import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;
import net.sourceforge.cilib.util.Cloneable;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.pso.particle.Particle;

/**
 * This interface augments {@link EnvironmentChangeResponseStrategy} to provide a 
 * particle specific environment response method, for when the particle itself
 * represents an environment, for example, heterogeneous architecture NN particles.
 */
public interface ParticleBasedEnvironmentChangeResponseStrategy {
    
    /**
     * Method the must be implemented in order to respond to a change
     * in a particle specific environment.
     * @param algorithm The algorithm optimizing the entity
     * @param entity The entity/particle for that is being responded to
     */
    // public <E extends HasTopology & Algorithm & HasNeighbourhood> void respond(E algorithm, Entity entity);
    public <P extends Particle, A extends SinglePopulationBasedAlgorithm<P>> void respond(A algorithm, Entity entity);
}
