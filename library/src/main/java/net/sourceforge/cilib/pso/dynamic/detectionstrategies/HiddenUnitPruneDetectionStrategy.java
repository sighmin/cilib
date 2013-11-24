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
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.nn.architecture.visitors.OutputErrorVisitor;
import net.sourceforge.cilib.problem.nn.NNDataTrainingProblem;
import net.sourceforge.cilib.problem.nn.EntitySpecificNNSlidingWindowTrainingProblem;

import java.util.List;
import java.util.LinkedList;

/**
 * This class implements {@link ParticleBasedEnvironmentChangeDetectionStrategy} 
 * by detecting changes on a per entity basis, that is, the environment is specific
 * to the entity.
 */
public class HiddenUnitPruneDetectionStrategy extends EnvironmentChangeDetectionStrategy {

    public HiddenUnitPruneDetectionStrategy() {
    }

    public HiddenUnitPruneDetectionStrategy(EnvironmentChangeDetectionStrategy rhs) {
        super(rhs);
    }

    @Override
    public HiddenUnitPruneDetectionStrategy getClone() {
        return new HiddenUnitPruneDetectionStrategy(this);
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm, Entity entity) {
        if (algorithm.getIterations() != 0 && algorithm.getIterations() % interval == 0) {
            // return if NN is already at it's smallest, and overfitting is a result of something other than too many hidden units
            int num_hidden = ((HeterogeneousNNChargedParticle) entity).getNumHiddenUnits().intValue();
            if (num_hidden < 2){
                return false;
            }

            // Fix me: implement Robel's overfitting detection condition!!!
            // Fix me: implement Ev trend detector condition!!!

            return true;
        }
        return false;
    }
}
