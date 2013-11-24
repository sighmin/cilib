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
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;

import java.util.List;
import java.util.LinkedList;

/**
 * This class implements {@link ParticleBasedEnvironmentChangeDetectionStrategy} 
 * by detecting changes on a per entity basis, that is, the environment is specific
 * to the entity.
 */
public class HiddenUnitPruneDetectionStrategy extends EnvironmentChangeDetectionStrategy {

    private ControlParameter validationGrowSensitivity;

    public HiddenUnitPruneDetectionStrategy() {
        this.validationGrowSensitivity = ConstantControlParameter.of(1.2);
    }

    public HiddenUnitPruneDetectionStrategy(HiddenUnitPruneDetectionStrategy rhs) {
        super(rhs);
        this.validationGrowSensitivity = rhs.validationGrowSensitivity.getClone();
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

            EntitySpecificNNSlidingWindowTrainingProblem problem = (EntitySpecificNNSlidingWindowTrainingProblem) algorithm.getOptimisationProblem();
            HeterogeneousNNChargedParticle particle = (HeterogeneousNNChargedParticle) entity;
            boolean robels_overfit = false;
            boolean ev_jump        = false;

            // Robels generlisation factor for overfitting test
            LinkedList<Double> robelsList = ((LinkedList<Double>)particle.getRobelsFactorList());
            double robels_factor = robelsList.peekLast();
            double robels_avg_plus_stdev = particle.getAverageMovingRobelsFactor() + particle.getSTDEVRobelsFactor();
            double robels_previous_phi = particle.getPreviousPhi();
            double phi = Math.min( Math.min(robels_previous_phi, robels_avg_plus_stdev), 1.0 );

            if (robels_factor > phi){
                robels_overfit = true;
            }

            // Ev increase detection logic
            double validationAvg   = particle.getAverageMovingOverfittingValidationError();
            double validationSTDEV = particle.getSTDEVOverfittingValidationError();
            double validationError = problem.getMSEValidationError(entity);
            if (validationError > validationAvg + (validationGrowSensitivity.getParameter() * validationSTDEV)){
                ev_jump = true;
            }

            boolean overfitting = (robels_overfit && ev_jump); // play with this to test conditions individually first
            return overfitting;
        }
        return false;
    }
}
