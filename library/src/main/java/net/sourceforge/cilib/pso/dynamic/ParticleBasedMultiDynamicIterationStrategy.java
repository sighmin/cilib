/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic;

import net.sourceforge.cilib.algorithm.population.IterationStrategy;
import net.sourceforge.cilib.problem.boundaryconstraint.BoundaryConstraint;
import net.sourceforge.cilib.pso.PSO;
import net.sourceforge.cilib.pso.dynamic.detectionstrategies.EnvironmentChangeDetectionStrategy;
import net.sourceforge.cilib.pso.dynamic.responsestrategies.EnvironmentChangeResponseStrategy;
import net.sourceforge.cilib.pso.iterationstrategies.SynchronousIterationStrategy;
import net.sourceforge.cilib.entity.Entity;
import java.util.LinkedList;
import java.util.List;

/**
 * Multi-Dynamic iteration strategy for PSO in dynamic environments.
 * In each iteration, it checks for several environmental changes, then
 * invokes the appropriate response strategy to deal with the change.
 */
public class ParticleBasedMultiDynamicIterationStrategy implements IterationStrategy<PSO> {

    private IterationStrategy<PSO> iterationStrategy;
    private List<EnvironmentChangeDetectionStrategy> detectionStrategies;
    private List<EnvironmentChangeResponseStrategy> responseStrategies;

    /**
     * Create a new instance of {@linkplain ParticleBasedMultiDynamicIterationStrategy}.
     */
    public ParticleBasedMultiDynamicIterationStrategy() {
        this.iterationStrategy = new SynchronousIterationStrategy();
        this.detectionStrategies = new LinkedList<EnvironmentChangeDetectionStrategy>();
        this.responseStrategies = new LinkedList<EnvironmentChangeResponseStrategy>();
    }

    /**
     * Create a copy of the provided instance.
     * @param copy The instance to copy.
     */
    public ParticleBasedMultiDynamicIterationStrategy(ParticleBasedMultiDynamicIterationStrategy copy) {
        this.iterationStrategy = copy.iterationStrategy.getClone();
        this.detectionStrategies = new LinkedList<EnvironmentChangeDetectionStrategy>();
        this.responseStrategies = new LinkedList<EnvironmentChangeResponseStrategy>();
        for (int i = 0; i < detectionStrategies.size(); ++i){
            this.detectionStrategies.set(i, copy.detectionStrategies.get(i).getClone());
            this.responseStrategies.set(i, copy.responseStrategies.get(i).getClone());
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ParticleBasedMultiDynamicIterationStrategy getClone() {
        return new ParticleBasedMultiDynamicIterationStrategy(this);
    }

    /**
     * Structure of Dynamic iteration strategy with re-initialisation:
     *
     * <ol>
     *   <li>Check for environment changes</li>
     *   <li>If the environment has changed:</li>
     *   <ol>
     *     <li>Respond to change with appropriate response</li>
     *   <ol>
     *   <li>Perform normal iteration</li>
     * </ol>
     */
    @Override
    public void performIteration(PSO algorithm) {
        fj.data.List<? extends Entity> particles = algorithm.getTopology();

        // Update particles error trends
        for (Entity particle : particles){
            ((HeterogeneousNNChargedParticle) particle).updateErrorTrends(algorithm.getOptimisationProblem());
        }

        // Iterate detection & response strategies (Grow & Prune NN's if conditions fired)
        for (Entity particle : particles){
            for (int i = 0; i < detectionStrategies.size(); ++i){
                try {
                    if (detectionStrategies.get(i).detect(algorithm, particle)){
                        responseStrategies.get(i).respond(algorithm, particle);
                    }
                } catch (UnsupportedOperationException e){ // attempt to run
                    if (detectionStrategies.get(i).detect(algorithm)){
                        responseStrategies.get(i).respond(algorithm);
                    }
                }
            }
        }

        iterationStrategy.performIteration(algorithm);
    }

    /**
     * Get the current {@linkplain IterationStrategy}.
     * @return The current {@linkplain IterationStrategy}.
     */
    public IterationStrategy<PSO> getIterationStrategy() {
        return iterationStrategy;
    }

    /**
     * Set the {@linkplain IterationStrategy} to be used.
     * @param iterationStrategy The value to set.
     */
    public void setIterationStrategy(IterationStrategy<PSO> iterationStrategy) {
        this.iterationStrategy = iterationStrategy;
    }

    /**
     * Get the currently defined {@linkplain EnvironmentChangeDetectionStrategy}.
     * @return The current {@linkplain EnvironmentChangeDetectionStrategy}.
     */
    public List<EnvironmentChangeDetectionStrategy> getDetectionStrategies() {
        return detectionStrategies;
    }

    /**
     * Set the {@linkplain EnvironmentChangeDetectionStrategy} to be used.
     * @param detectionStrategies The {@linkplain EnvironmentChangeDetectionStrategy} to set.
     */
    public void setDetectionStrategy(List<EnvironmentChangeDetectionStrategy> detectionStrategies) {
        this.detectionStrategies = detectionStrategies;
    }

    /**
     * Add a detection strategy to the list of detection strategies.
     * @param strategy The {@linkplain EnvironmentChangeDetectionStrategy} to add.
     */
    public void addDetectionStrategy(EnvironmentChangeDetectionStrategy strategy){
        this.detectionStrategies.add(strategy);
    }

    /**
     * Get the currently defined {@linkplain EnvironmentChangeResponseStrategy},
     * @return The current {@linkplain EnvironmentChangeResponseStrategy}.
     */
    public List<EnvironmentChangeResponseStrategy> getResponseStrategy() {
        return responseStrategies;
    }

    /**
     * Set the current {@linkplain EnvironmentChangeResponseStrategy} to use.
     * @param responseStrategies The {@linkplain EnvironmentChangeResponseStrategy} to set.
     */
    public void setResponseStrategy(List<EnvironmentChangeResponseStrategy> responseStrategies) {
        this.responseStrategies = responseStrategies;
    }

    /**
     * Add a response strategy to the list of response strategies.
     * @param strategy The {@linkplain EnvironmentChangeResponseStrategy} to add.
     */
    public void addResponseStrategy(EnvironmentChangeResponseStrategy strategy){
        this.responseStrategies.add(strategy);
    }

    @Override
    public BoundaryConstraint getBoundaryConstraint() {
        return this.iterationStrategy.getBoundaryConstraint();
    }

    @Override
    public void setBoundaryConstraint(BoundaryConstraint boundaryConstraint) {
        this.iterationStrategy.setBoundaryConstraint(boundaryConstraint);
    }
}
