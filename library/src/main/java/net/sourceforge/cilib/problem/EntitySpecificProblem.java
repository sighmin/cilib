/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.problem;

import net.sourceforge.cilib.problem.solution.Fitness;
import net.sourceforge.cilib.type.DomainRegistry;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.util.Cloneable;
import net.sourceforge.cilib.entity.Entity;

/**
 * Optimisation problems are characterized by a domain that specifies the search
 * space and a fitness given a potential solution. This interface ensures that
 * an {@linkplain net.sourceforge.cilib.algorithm.Algorithm} has all
 * the information it needs to find a solution to a given optimisation problem.
 * In addition, it is the responsibility of an optimisation problem to keep
 * track of the number of times the fitness has been evaluated.
 * <p>
 * All problems that have entity specific fitness calculations must implement this interface.
 */
public interface EntitySpecificProblem {

    /**
     * Calculates the fitness on a per entity basis. 
     *
     * @param entity  the potential solution found by the optimisation algorithm.
     * @return          the fitness of the solution.
     */
    public Fitness getFitness(Entity entity);

}
