/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.util.calculator;

import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.algorithm.Algorithm;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.problem.solution.Fitness;
import net.sourceforge.cilib.problem.nn.EntitySpecificNNSlidingWindowTrainingProblem;

/**
 * A fitness calculator that is specialised to determine the fitness of
 * an Entity instance.
 */
public class EntitySpecificFitnessCalculator implements FitnessCalculator<Entity> {
    private static final long serialVersionUID = -5053760817332028741L;

    /**
     * {@inheritDoc}
     */
    @Override
    public EntitySpecificFitnessCalculator getClone() {
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Fitness getFitness(Entity entity) {
        Algorithm algorithm = AbstractAlgorithm.get();
        // To do: fix this code smell
        return ((EntitySpecificNNSlidingWindowTrainingProblem) algorithm.getOptimisationProblem()).getFitness(entity);
    }

}
