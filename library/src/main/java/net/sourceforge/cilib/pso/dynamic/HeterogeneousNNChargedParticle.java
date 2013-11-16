/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic;

import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.problem.Problem;
import net.sourceforge.cilib.problem.nn.NNTrainingProblem;
import net.sourceforge.cilib.problem.solution.InferiorFitness;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * Charged Particle used by charged PSO (ChargedVelocityProvider). The only
 * difference from DynamicParticle is that a charged particle stores the charge
 * magnitude and the initialisation strategy for charge.
 *
 *
 */
public class HeterogeneousNNChargedParticle extends ChargedParticle {

    public HeterogeneousNNChargedParticle() {

    }

    public HeterogeneousNNChargedParticle(HeterogeneousNNChargedParticle copy) {
        super(copy);
    }

    @Override
    public HeterogeneousNNChargedParticle getClone() {
        return new HeterogeneousNNChargedParticle(this);
    }

    @Override
    public void initialise(Problem problem) {
        super.initialise(problem);

        // init NN parameters
        //this.getProperties().put(EntityType.Particle.VELOCITY, Vector.copyOf((Vector) getCandidateSolution()));

        // hidden units
        int num_hidden = ((NNTrainingProblem) problem).getNeuralNetwork().
            getArchitecture().
            getArchitectureBuilder().
            getLayerConfigurations().
            get(1).getSize();
        this.getProperties().put(EntityType.HeteroNN.NUM_HIDDEN, Int.valueOf(num_hidden));
        this.getProperties().put(EntityType.HeteroNN.BITMASK, Vector.fill(Int.valueOf(1), getCandidateSolution().size()) );
    }

    /**
     * @return the number of hidden units of this particle's architecture
     */
    public Int getNumHiddenUnits(){
        return (Int) this.getProperties().get(EntityType.HeteroNN.NUM_HIDDEN);
    }

    /**
     * @return the bitmask indicating relevant dimensions of the solution vector
     */
    public Vector getBitMask(){
        return (Vector) this.getProperties().get(EntityType.HeteroNN.BITMASK);
    }

    /**
     * Builds a weight vector with only this particles architectures relevant weights
     * @return the built weight vector
     */
    public Vector getWeightVector(){
        // create architecture specific weight vector
        Vector maskedSolution = (Vector) this.getCandidateSolution();
        Vector mask = (Vector) this.getBitMask();
        Vector.Builder weightVectorBuilder = Vector.newBuilder();

        for (int i = 0; i < maskedSolution.size(); ++i){
            int bit = mask.get(i).intValue();
            if (bit == 1){
                weightVectorBuilder.add(maskedSolution.get(i));
            }
        }
        return weightVectorBuilder.build();
    }
}
