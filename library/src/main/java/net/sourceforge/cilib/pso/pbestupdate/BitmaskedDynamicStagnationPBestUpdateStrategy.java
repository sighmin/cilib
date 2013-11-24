/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.pbestupdate;

import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.problem.solution.Fitness;

/**
 * Update the personal best of the particle, as well as it's corresponding
 * bitmask. The stagnation counter is also only updated if a change is smaller
 * than some epsilon, since fitnesses cannot be exactly the same in dynamic
 * environments.
 */
public class BitmaskedDynamicStagnationPBestUpdateStrategy implements PersonalBestUpdateStrategy {

    private static final long serialVersionUID = 266386833476786081L;
    private ControlParameter epsilon = ConstantControlParameter.of(0.01);

    /**
     * {@inheritDoc}
     */
    @Override
    public PersonalBestUpdateStrategy getClone() {
        return this;
    }

    /**
     * If the current fitness is better than the current best fitness, update
     * the best fitness of the particle to equal the current fitness and make
     * the personal best position a clone of the current particle position, as
     * well as updating the best bitmask.
     *
     * If the current fitness is not better than the current best fitness by
     * some epsilon, increase the particle's pbest stagnation counter. This
     * method is more appropriate in dynamic environments.
     *
     * @param particle The particle to update.
     */
    @Override
    public void updatePersonalBest(Particle particle) {
        Fitness currentFitness = particle.getFitness();
        Fitness bestFitness = particle.getBestFitness();

        // PBest changed within the epsilon value
        double delta = Math.abs(  Math.abs(currentFitness.getValue().doubleValue()) - Math.abs(bestFitness.getValue().doubleValue())  );
        if (delta > epsilon.getParameter()) {
            particle.getProperties().put(EntityType.Particle.Count.PBEST_STAGNATION_COUNTER, Int.valueOf(0));
        } else {
            //PBest only changed within epsilon, so it not significant enough a change. Increment stagnation counter.
            int count = ((Int)particle.getProperties().get(EntityType.Particle.Count.PBEST_STAGNATION_COUNTER)).intValue();
            particle.getProperties().put(EntityType.Particle.Count.PBEST_STAGNATION_COUNTER,  Int.valueOf(++count));
        }

        // PBest changed
        if (currentFitness.compareTo(bestFitness) > 0) {
            particle.getParticleBehavior().incrementSuccessCounter();
            particle.getProperties().put(EntityType.HeteroNN.BEST_BITMASK, particle.getProperties().get(EntityType.HeteroNN.BITMASK));
            particle.getProperties().put(EntityType.HeteroNN.BEST_NUM_HIDDEN, particle.getProperties().get(EntityType.HeteroNN.NUM_HIDDEN));
            particle.getProperties().put(EntityType.Particle.BEST_FITNESS, particle.getFitness());
            particle.getProperties().put(EntityType.Particle.BEST_POSITION, particle.getCandidateSolution().getClone());
            return;
        }
    }

    public void setEpsilon(double epsilon){
        this.epsilon = ConstantControlParameter.of(epsilon);
    }
}
