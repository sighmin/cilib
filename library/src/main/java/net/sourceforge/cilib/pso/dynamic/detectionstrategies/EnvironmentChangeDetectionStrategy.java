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
 * This abstract class defines the interface that detection strategies have to adhere to.
 * Detection strategies are used within the scope of a {@link DynamicIterationStrategy} to
 * detect whether the environment has change during the course of an
 * {@link Algorithm algorithm's} execution.
 */
public abstract class EnvironmentChangeDetectionStrategy implements Cloneable, ParticleBasedEnvironmentChangeDetectionStrategy {
    protected double epsilon = 0.0;
    protected int interval = 0;

    public EnvironmentChangeDetectionStrategy() {
        epsilon = 0.001;
        interval = 10;
    }

    public EnvironmentChangeDetectionStrategy(EnvironmentChangeDetectionStrategy rhs) {
        epsilon = rhs.epsilon;
        interval = rhs.interval;
    }

    /**
     * Clone the <tt>EnvironmentChangeDetectionStrategy</tt> object.
     * @return A cloned <tt>EnvironmentChangeDetectionStrategy</tt>
     */
    public abstract EnvironmentChangeDetectionStrategy getClone();

    /**
     * Check the environment in which the specified PSO algorithm is running for changes.
     * @param algorithm The <tt>PSO</tt> that runs in a dynamic environment.
     * @return true if any changes are detected, false otherwise
     */
    public abstract <E extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(E algorithm);

    /**
     * Check the environment in which the specific entities environment changes.
     * @param algorithm The <tt>PSO</tt> that runs in a dynamic environment.
     * @param entity The <tt>Entity</tt> containing the environment to test.
     * @return true if any changes are detected, false otherwise
     */
    public <E extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(E algorithm, Entity entity){
        throw new UnsupportedOperationException("Not implemented in sub class of EnvironmentChangeDetectionStrategy.");
    }

    /**
     * Sets epsilon, the environment change significance indicator.
     * @param e The value of epsilon
     */
    public void setEpsilon(double e) {
        if (e < 0.0) {
            throw new IllegalArgumentException("The epsilon value cannot be negative");
        }

        epsilon = e;
    }

    /**
     * Gets epsilon, the environment change significance indicator.
     * @return epsilon The change significance indicator
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Sets interval, the frequency (in number of iterations) that the detection condition is tested.
     * @param interval The frequency
     */
    public void setInterval(int interval) {
        if (interval <= 0) {
            throw new IllegalArgumentException("The number of consecutive iterations to pass cannot be <= 0");
        }
        this.interval = interval;
    }

    /**
     * Sets interval, the frequency (in number of iterations) that the detection condition is tested.
     * @param interval The frequency
     */
    public void setIterationsModulus(int im) {
        if (im <= 0) {
            throw new IllegalArgumentException("The number of consecutive iterations to pass cannot be <= 0");
        }

        interval = im;
    }

    /**
     * Gets the frequency of change detection, interval
     * @return interval The frequency
     */
    public int getIterationsModulus() {
        return interval;
    }
}
