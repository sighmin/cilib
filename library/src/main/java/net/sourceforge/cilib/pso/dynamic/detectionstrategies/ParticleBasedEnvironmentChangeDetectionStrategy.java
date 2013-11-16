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
 * This abstract class defines the interface that particle based detection strategies have to adhere to.
 * Detection strategies are used within the scope of a {@link DynamicIterationStrategy} to
 * detect whether the environment has change during the course of an
 * {@link Algorithm algorithm's} execution.
 * 
 * This class also exposes a particle based detection method where the environment is
 * specific to the particle.
 */
public abstract class ParticleBasedEnvironmentChangeDetectionStrategy implements Cloneable {
    protected double epsilon;
    protected int interval;

    public ParticleBasedEnvironmentChangeDetectionStrategy() {
        this.epsilon = 0.001;
        this.interval = 1;
    }

    public ParticleBasedEnvironmentChangeDetectionStrategy(ParticleBasedEnvironmentChangeDetectionStrategy rhs) {
        this.epsilon = rhs.epsilon;
        this.interval = rhs.interval;
    }

    /**
     * Clone the <tt>ParticleBasedEnvironmentChangeDetectionStrategy</tt> object.
     * @return A cloned <tt>ParticleBasedEnvironmentChangeDetectionStrategy</tt>
     */
    public abstract ParticleBasedEnvironmentChangeDetectionStrategy getClone();

    /**
     * Check the environment in which the specified PSO algorithm is running for changes.
     * @param algorithm The <tt>PSO</tt> that runs in a dynamic environment.
     * @param entity The entity for which the environment is specific.
     * @return true if any changes are detected, false otherwise.
     */
    public abstract <E extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(E algorithm, Entity entity);

    public void setEpsilon(double epsilon) {
        if (epsilon < 0.0) {
            throw new IllegalArgumentException("The epsilon value cannot be negative");
        }
        this.epsilon = epsilon;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setInterval(int interval) {
        if (interval <= 0) {
            throw new IllegalArgumentException("The number of consecutive iterations to pass cannot be <= 0");
        }
        this.interval = interval;
    }

    public int getIterationsModulus() {
        return interval;
    }
}
