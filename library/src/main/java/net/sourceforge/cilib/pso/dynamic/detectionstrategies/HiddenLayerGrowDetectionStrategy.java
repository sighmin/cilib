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
import net.sourceforge.cilib.problem.nn.EntitySpecificNNSlidingWindowTrainingProblem;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.type.types.Int;

/**
 * This class implements {@link ParticleBasedEnvironmentChangeDetectionStrategy} 
 * by detecting changes on a per entity basis, that is, the environment is specific
 * to the entity.
 */
public class HiddenLayerGrowDetectionStrategy extends EnvironmentChangeDetectionStrategy {

    private ControlParameter acceptableClassificationError;
    private ControlParameter errorSpikeSensitivity;
    private ControlParameter windowSize;

    public HiddenLayerGrowDetectionStrategy() {
        this.windowSize = ConstantControlParameter.of(5);
        this.acceptableClassificationError = ConstantControlParameter.of(0.15);
        this.errorSpikeSensitivity = ConstantControlParameter.of(2.0);
    }

    public HiddenLayerGrowDetectionStrategy(HiddenLayerGrowDetectionStrategy rhs) {
        super(rhs);
        this.windowSize = rhs.windowSize.getClone();
        this.acceptableClassificationError = rhs.acceptableClassificationError.getClone();
        this.errorSpikeSensitivity = rhs.errorSpikeSensitivity.getClone();
    }

    @Override
    public HiddenLayerGrowDetectionStrategy getClone() {
        return new HiddenLayerGrowDetectionStrategy(this);
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm, Entity entity) {
        if (algorithm.getIterations() != 0 && algorithm.getIterations() % interval == 0) {
            HeterogeneousNNChargedParticle particle = (HeterogeneousNNChargedParticle) entity;
            EntitySpecificNNSlidingWindowTrainingProblem problem = (EntitySpecificNNSlidingWindowTrainingProblem) algorithm.getOptimisationProblem();

            // calculate errorSpike condition
            boolean errorSpike = false;
            double trainingAvg   = particle.getAverageMovingTrainingError();
            double trainingSTDEV = particle.getSTDEVMovingTrainingError();
            double trainingError = problem.getMSETrainingError(entity);
            if (trainingError > trainingAvg + (errorSpikeSensitivity.getParameter() * trainingSTDEV)){
                errorSpike = true;
                System.out.println("error spike");
            }

            // calculate stagnation with unacceptable error conditions
            boolean stagnationWithUnacceptableError = false;
            boolean stagnation = false;
            double classificationValidationError = problem.getClassificationValidationError(entity);
            int counter = ((Int)entity.getProperties().get(EntityType.Particle.Count.PBEST_STAGNATION_COUNTER)).intValue();
            if (counter > windowSize.getParameter()) {
                entity.getProperties().put(EntityType.Particle.Count.PBEST_STAGNATION_COUNTER, Int.valueOf(0));
                stagnation = true;
            }
            if (stagnation && classificationValidationError > acceptableClassificationError.getParameter()){
                stagnationWithUnacceptableError = true;
                System.out.println("stagnation with unacceptable error");
            }

            // make decision
            if (errorSpike || stagnationWithUnacceptableError){
                return true;
            }
        }
        return false;
    }

    public void setAcceptableClassificationError(ControlParameter acceptableClassificationError){
        this.acceptableClassificationError = acceptableClassificationError;
    }

    public void setErrorSpikeSensitivity(ControlParameter errorSpikeSensitivity){
        this.errorSpikeSensitivity = errorSpikeSensitivity;
    }

    public void setWindowSize(ControlParameter windowSize) {
        this.windowSize = windowSize;
    }
}
