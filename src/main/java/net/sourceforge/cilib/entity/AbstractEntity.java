/*
 * Copyright (C) 2003 - 2008
 * Computational Intelligence Research Group (CIRG@UP)
 * Department of Computer Science
 * University of Pretoria
 * South Africa
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
package net.sourceforge.cilib.entity;

import java.util.Comparator;
import net.sourceforge.cilib.entity.comparator.AscendingFitnessComparator;
import net.sourceforge.cilib.entity.comparator.DescendingFitnessComparator;
import net.sourceforge.cilib.problem.Fitness;
import net.sourceforge.cilib.problem.MaximisationFitness;
import net.sourceforge.cilib.problem.MinimisationFitness;
import net.sourceforge.cilib.pso.positionupdatestrategies.IterationNeighbourhoodBestUpdateStrategy;
import net.sourceforge.cilib.pso.positionupdatestrategies.NeighbourhoodBestUpdateStrategy;
import net.sourceforge.cilib.type.types.Blackboard;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.util.calculator.FitnessCalculator;
import net.sourceforge.cilib.util.calculator.VectorBasedFitnessCalculator;

/**
 * Abstract class definition for all concrete {@linkplain Entity} objects.
 * This class defines the {@linkplain Entity} main data structure for the
 * values stored within the {@linkplain Entity} itself.
 */
public abstract class AbstractEntity implements Entity, CandidateSolution {
	private static final long serialVersionUID = 3104817182593047611L;

	private long id;
	private final CandidateSolution candidateSolution;
	protected NeighbourhoodBestUpdateStrategy neighbourhoodBestUpdateStrategy;
	private FitnessCalculator fitnessCalculator;

	/**
	 * Initialise the candidate solution of the {@linkplain Entity}.
	 */
	protected AbstractEntity() {
		this.id = EntityIdFactory.getNextId();
		
		this.candidateSolution = new CandidateSolutionMixin();
		this.neighbourhoodBestUpdateStrategy = new IterationNeighbourhoodBestUpdateStrategy();
		this.fitnessCalculator = new VectorBasedFitnessCalculator();
	}
	
	/**
	 * Copy constructor. Instantiate and copy the given instance. 
	 * @param copy The instance to copy.
	 */
	protected AbstractEntity(AbstractEntity copy) {
		this.id = EntityIdFactory.getNextId();
		
		this.candidateSolution = (CandidateSolution) copy.candidateSolution.getClone();
		this.neighbourhoodBestUpdateStrategy = copy.neighbourhoodBestUpdateStrategy.getClone();
		this.fitnessCalculator = copy.fitnessCalculator.getClone();
	}
	
	/**
	 * {@inheritDoc}
	 *
	 * @param object The object to compare equality.
	 */
	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		
		if ((object == null) || (this.getClass() != object.getClass()))
			return false;
		
		AbstractEntity other = (AbstractEntity) object;
		return (this.id == other.id) && (this.candidateSolution.equals(other.candidateSolution));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int hashCode() {
		int hash = 7;
		hash = 31 * hash + (int)(id ^ (id >>> 32));
		hash = 31 * hash + (this.candidateSolution == null ? 0 : this.candidateSolution.hashCode());
		return hash;
	}
	
	/**
	 * Get the properties associate with the <code>Entity</code>.
	 * @return The properties within a {@linkplain Blackboard}.
	 */
	@Override
	public final Blackboard<Enum<?>, Type> getProperties() {
		return this.candidateSolution.getProperties();
	}

	/**
	 * Set the properties for the current <code>Entity</code>.
	 * @param properties The {@linkplain Blackboard} containing the new properties.
	 */
	@Override
	public final void setProperties(Blackboard<Enum<?>, Type> properties) {
		this.candidateSolution.setProperties(properties);
	}

	/**
	 * Get the value of the {@linkplain CandidateSolution} maintained by this 
	 * {@linkplain Entity}.
	 * @return The candidate solution as a {@linkplain Type}.
	 */
	@Override
	public Type getCandidateSolution() {
		return this.candidateSolution.getCandidateSolution();
	}

	/**
	 * Get the fitness of the {@linkplain CandidateSolution} maintained by this 
	 * {@linkplain Entity}.
	 * @return The {@linkplain Fitness} of the candidate solution.
	 */
	@Override
	public Fitness getFitness() {
		return this.candidateSolution.getFitness();
	}

	/**
	 * Set the {@linkplain Type} maintained by this {@linkplain Entity}s
	 * {@linkplain CandidateSolution}.
	 * @param candidateSolution The {@linkplain Type} that will be the new value of the
	 *        {@linkplain Entity} {@linkplain CandidateSolution}.
	 */
	@Override
	public void setCandidateSolution(Type candidateSolution) {
		this.candidateSolution.setCandidateSolution(candidateSolution);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public final Fitness getSocialBestFitness() {
		return this.neighbourhoodBestUpdateStrategy.getSocialBestFitness(this);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public Fitness getBestFitness() {
		return getFitness();
	}
	
	/**
	 * Get the reference to the currently employed <code>NeighbourhoodBestUpdateStrategy</code>.
	 * @return A reference to the current <code>NeighbourhoodBestUpdateStrategy</code> object
	 */
	public NeighbourhoodBestUpdateStrategy getNeighbourhoodBestUpdateStrategy() {
		return this.neighbourhoodBestUpdateStrategy;
	}
	
	/**
	 * Set the <code>NeighbourhoodBestUpdateStrategy</code> to be used by the {@linkplain Entity}.
	 * @param neighbourhoodBestUpdateStrategy The <code>NeighbourhoodBestUpdateStrategy</code> to be used
	 */
	public void setNeighbourhoodBestUpdateStrategy(NeighbourhoodBestUpdateStrategy neighbourhoodBestUpdateStrategy) {
		this.neighbourhoodBestUpdateStrategy = neighbourhoodBestUpdateStrategy;
	}

	/**
	 * Get the current {@code FitnessCalculator} for the current {@code Entity}.
	 * @return The {@code FitnessCalculator} associated with this {@code Entity}.
	 */
	public FitnessCalculator getFitnessCalculator() {
		return fitnessCalculator;
	}

	/**
	 * Set the {@code FitnessCalculator} for the current {@code Entity}.
	 * @param fitnessCalculator The value to set.
	 */
	public void setFitnessCalculator(FitnessCalculator fitnessCalculator) {
		this.fitnessCalculator = fitnessCalculator;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public long getId() {
		return this.id;
	}

	@Override
	public final Comparator<Entity> getComparator() {
		return (getFitness() instanceof MinimisationFitness) ?
			new AscendingFitnessComparator() :
			new DescendingFitnessComparator();
	}

}
