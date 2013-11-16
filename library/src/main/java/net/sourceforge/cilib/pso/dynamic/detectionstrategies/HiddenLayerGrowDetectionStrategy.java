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

/**
 * This class implements {@link ParticleBasedEnvironmentChangeDetectionStrategy} 
 * by detecting changes on a per entity basis, that is, the environment is specific
 * to the entity.
 */
public class HiddenLayerGrowDetectionStrategy extends EnvironmentChangeDetectionStrategy {
    private static final long serialVersionUID = 4079212153655661164L;

    public HiddenLayerGrowDetectionStrategy() {
    }

    public HiddenLayerGrowDetectionStrategy(EnvironmentChangeDetectionStrategy rhs) {
        super(rhs);
    }

    @Override
    public HiddenLayerGrowDetectionStrategy getClone() {
        return new HiddenLayerGrowDetectionStrategy(this);
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm) {
        return true;
    }

    @Override
    public <A extends HasTopology & Algorithm & HasNeighbourhood> boolean detect(A algorithm, Entity entity) {
        if (algorithm.getIterations() != 0 && algorithm.getIterations() % interval == 0) {
            boolean grow = false;

            // detect grow conditions here

            // get problem for fitness from within algorithm object
            // get partitioned data sets from problem

            // if (grow == true){
            //     return true;
            // } else {
            //     return false;
            // }
            // System.out.println("detected grow!");
            return true;
        }
        return false;
    }
}
