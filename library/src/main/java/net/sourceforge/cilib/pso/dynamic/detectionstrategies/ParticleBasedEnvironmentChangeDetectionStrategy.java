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
import net.sourceforge.cilib.pso.dynamic.DynamicIterationStrategy;
import net.sourceforge.cilib.util.Cloneable;
import net.sourceforge.cilib.entity.Entity;

/**
 * This interface augments {@link EnvironmentChangeDetectionStrategy} to provide a 
 * particle specific environment detection method, for when the particle itself
 * represents an environment, for example, heterogeneous architecture NN particles.
 */
public interface ParticleBasedEnvironmentChangeDetectionStrategy {

    /**
     * Method the must be implemented in order to detect a change in a 
     * particle specific environment.
     * @param algorithm The algorithm optimizing the entity
     * @param entity The entity/particle for that is being tested
     * @return a boolean value whether a change was detected or not
     */
    public <E extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(E algorithm, Entity entity);
}
