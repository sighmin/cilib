/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.problem.nn;

import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.io.DataTable;
import net.sourceforge.cilib.io.DataTableBuilder;
import net.sourceforge.cilib.io.DelimitedTextFileReader;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.exception.CIlibIOException;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.io.transform.ShuffleOperator;
import net.sourceforge.cilib.io.transform.TypeConversionOperator;
import net.sourceforge.cilib.nn.NeuralNetworks;
import net.sourceforge.cilib.nn.architecture.visitors.OutputErrorVisitor;
import net.sourceforge.cilib.problem.AbstractProblem;
import net.sourceforge.cilib.problem.solution.Fitness;
import net.sourceforge.cilib.problem.EntitySpecificProblem;
import net.sourceforge.cilib.type.DomainRegistry;
import net.sourceforge.cilib.type.StringBasedDomainRegistry;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.Int;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.pso.dynamic.HeterogeneousNNChargedParticle;

/**
 * Class represents a {@link NNTrainingProblem} where the goal is to optimize
 * the set of weights of a neural network to best fit a given dynamic dataset (either
 * regression, classification etc.). Sliding window is used to simulate dynamic changes.
 * User-specified step size, frequency, and sliding window size control the dynamics
 * of the sliding window. Sliding window moves over the dataset and presents patterns
 * to the neural network in batches equal to the size of the window.
 */
public class EntitySpecificNNSlidingWindowTrainingProblem extends NNTrainingProblem implements EntitySpecificProblem {

    private DataTableBuilder dataTableBuilder;
    private DataTable dataTable; // stores the entire data set from which training & generalisation sets are sampled
    private int previousShuffleIteration;
    private int previousIteration;
    private boolean initialised;

    private int dataChangesCounter = 1; // # times the dataset was dynamically updated (has to start with 1)
    private int stepSize; // step size for each set, i.e. # patterns by which the sliding window moves forward in each dynamic step
    private int changeFrequency; // # algorithm iterations after which the window will slide
    private int windowSize; // number of patterns in the active set

    private double classificationSensitivityThreshold = 0.2;

    /**
     * Default constructor.
     */
    public EntitySpecificNNSlidingWindowTrainingProblem() {
        super();
        dataTableBuilder = new DataTableBuilder(new DelimitedTextFileReader());
        previousShuffleIteration = -1;
        previousIteration = -1;
        initialised = false;
    }

    /**
     * Initialises the problem by reading in the data and constructing the datatable,
     * as well as the initial training and generalisation sets. Also initialises (constructs) the neural network.
     */
    @Override
    public void initialise() {
        if (initialised) {
            return;
        }
        try {
            dataTableBuilder.addDataOperator(new TypeConversionOperator());
            dataTableBuilder.addDataOperator(patternConversionOperator);
            dataTableBuilder.buildDataTable();
            dataTable = dataTableBuilder.getDataTable();
            
            int trainingSize = (int) (windowSize * trainingSetPercentage);
            int validationSize = (int) (windowSize * validationSetPercentage);
            int generalisationSize = windowSize - trainingSize - validationSize;

            StandardPatternDataTable candidateSet = new StandardPatternDataTable();
            trainingSet = new StandardPatternDataTable();
            generalisationSet = new StandardPatternDataTable();
            validationSet = new StandardPatternDataTable();

            for (int i = 0; i < windowSize; i++) { // fetch patterns to fill the initial window
                candidateSet.addRow((StandardPattern) dataTable.removeRow(0));
            }

            ShuffleOperator initialShuffler = new ShuffleOperator();
            initialShuffler.operate(candidateSet);

            for (int i = 0; i < trainingSize; i++) {
                trainingSet.addRow((StandardPattern) candidateSet.getRow(i));
            }

            for (int i = trainingSize; i < validationSize + trainingSize; i++) {
                validationSet.addRow((StandardPattern) candidateSet.getRow(i));
            }

            for (int i = validationSize + trainingSize; i < generalisationSize + validationSize + trainingSize; i++) {
                generalisationSet.addRow((StandardPattern) candidateSet.getRow(i));
            }

            neuralNetwork.initialise();
        } catch (CIlibIOException exception) {
            exception.printStackTrace();
        }
        initialised = true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public AbstractProblem getClone() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * Calculates the fitness of the given solution by setting the neural network
     * weights to the solution and evaluating the training set in order to calculate
     * the MSE (which is minimized). Also checks whether the window has to be slided,
     * and slides the window when necessary by adjusting the training and generalisation sets.
     * @param solution the weights representing a solution.
     * @return a new MinimisationFitness wrapping the MSE training error.
     */
    @Override
    protected Fitness calculateFitness(Type solution) {
        if (trainingSet == null) {
            this.initialise();
        }

        int currentIteration = AbstractAlgorithm.get().getIterations();
        if (currentIteration != previousShuffleIteration) {
            try {
                shuffler.operate(trainingSet);
            } catch (CIlibIOException exception) {
                exception.printStackTrace();
            }
        }

        this.moveTheWindow(currentIteration);

        neuralNetwork.setWeights((Vector) solution);

        double errorTraining = 0.0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : trainingSet) {
            Vector output = neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            for (Numeric real : error) {
                errorTraining += real.doubleValue() * real.doubleValue();
            }
        }
        errorTraining /= trainingSet.getNumRows() * error.size();

        return objective.evaluate(errorTraining);
    }

    /**
     * Calculates the fitness of the given solution by setting the neural network
     * weights to the solution and evaluating the training set in order to calculate
     * the MSE (which is minimized). Also checks whether the window has to be slided,
     * and slides the window when necessary by adjusting the training and generalisation sets.
     * @param entity the entity containing the entity specific NN architecture required to calc fitness
     * @return a new MinimisationFitness wrapping the MSE training error.
     */
    public Fitness getFitness(Entity entity) {
        // init
        if (trainingSet == null) {
            this.initialise();
        }

        // shuffle training data window
        int currentIteration = AbstractAlgorithm.get().getIterations();
        if (currentIteration != previousShuffleIteration) {
            try {
                shuffler.operate(trainingSet);
            } catch (CIlibIOException exception) {
                exception.printStackTrace();
            }
        }

        // move window
        this.moveTheWindow(currentIteration);

        // calculate error
        this.reInitArchitecture(entity);

        // calculate error
        double errorTraining = 0.0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : trainingSet) {
            Vector output = neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            for (Numeric real : error) {
                errorTraining += real.doubleValue() * real.doubleValue();
            }
        }
        errorTraining /= trainingSet.getNumRows() * error.size();

        return objective.evaluate(errorTraining);
    }

    /**
     * Private method to move the window along the data set when necessar
     * @param currentIteration the current iteration number
     */    
    private void moveTheWindow(int currentIteration){
        // move window
        if(currentIteration - changeFrequency * dataChangesCounter == 0 && currentIteration != previousIteration) { // update training & generalisation sets (slide the window)
            try {
                previousIteration = currentIteration;
                dataChangesCounter++;

                StandardPatternDataTable candidateSet = new StandardPatternDataTable();
                for (int i = 0; i < stepSize; i++) {
                    candidateSet.addRow((StandardPattern) dataTable.removeRow(0));
                }

                ShuffleOperator initialShuffler = new ShuffleOperator();
                initialShuffler.operate(candidateSet);

                int trainingStepSize = (int)(stepSize * trainingSetPercentage);
                int validationStepSize = (int)(stepSize * validationSetPercentage);
                int generalisationStepSize = stepSize - trainingStepSize - validationStepSize;

                for (int t = 0; t < trainingStepSize; t++){
                    trainingSet.removeRow(0);
                    trainingSet.addRow(candidateSet.removeRow(0));
                }

                for (int t = 0; t < validationStepSize; t++){
                    validationSet.removeRow(0);
                    validationSet.addRow(candidateSet.removeRow(0));
                }

                for (int t = 0; t < generalisationStepSize; t++){
                    generalisationSet.removeRow(0);
                    generalisationSet.addRow(candidateSet.removeRow(0));
                }
            } catch (CIlibIOException exception) {
                exception.printStackTrace();
            }
        }
    }

    private void reInitArchitecture(Entity entity){
        HeterogeneousNNChargedParticle particle = (HeterogeneousNNChargedParticle) entity;

        // rebuild architecture and set weights
        Numeric numHidden = particle.getNumHiddenUnits();
        Numeric currentNumHidden = Int.valueOf(neuralNetwork.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(1).getSize());
        if (numHidden.compareTo(currentNumHidden) != 0){
            neuralNetwork.getArchitecture().getArchitectureBuilder().getLayerConfigurations().get(1).setSize(numHidden.intValue());
            neuralNetwork.initialise();
        }
        neuralNetwork.setWeights(particle.getWeightVector());
    }

    /**
     * Gets the classification error for entity on the training set
     * @return the training set classification error
     */
    public double getClassificationTrainingError(Entity entity) {
        return getClassificationError(entity, trainingSet);
    }

    /**
     * Gets the classification error for entity on the generalisation set
     * @return the generalisation set classification error
     */
    public double getClassificationGeneralisationError(Entity entity) {
        return getClassificationError(entity, generalisationSet);
    }

    /**
     * Gets the validation error for entity on the training set
     * @return the validation set classification error
     */
    public double getClassificationValidationError(Entity entity) {
        return getClassificationError(entity, validationSet);
    }

    /**
     * Private method to calculate the classification error (% incorrect classified patterns)
     * for a given set of patterns.
     * @param entity The entity for which the NN architecture is specific
     * @param patterns The data table containing patterns to calculate the error on
     */  
    private double getClassificationError(Entity entity, StandardPatternDataTable patterns){
        this.reInitArchitecture(entity);

        int numberPatternsCorrect = 0;
        int numberPatternsIncorrect = 0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : patterns) {
            neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            boolean isCorrect = true;
            
            for (Numeric real : error) {
                if (Math.abs(real.doubleValue()) > this.classificationSensitivityThreshold) {
                    isCorrect = false;
                    break;
                }
            }
            if (isCorrect){
                numberPatternsCorrect++;
            } else {
                numberPatternsIncorrect++;
            }
        }

        double percentageIncorrect = (double) numberPatternsIncorrect / ((double) numberPatternsIncorrect + (double) numberPatternsCorrect);
        return percentageIncorrect;
    }

    /**
     * Get the MSE on the training set, using the entities specific architecture.
     * @return the training MSE
     */
    public double getMSETrainingError(Entity entity){
        return getMSE(entity, trainingSet);
    }

    /**
     * Get the MSE on the generalisation set, using the entities specific architecture.
     * @return the generalisation MSE
     */
    public double getMSEGeneralisationError(Entity entity){
        return getMSE(entity, generalisationSet);
    }

    /**
     * Get the MSE on the validation set, using the entities specific architecture.
     * @return the validation MSE
     */
    public double getMSEValidationError(Entity entity){
        return getMSE(entity, validationSet);
    }

    /**
     * Private method to calculate the MSE for a given set of patterns.
     * @param entity The entity for which the NN architecture is specific
     * @param patterns The data table containing patterns to calculate the error on
     */  
    private double getMSE(Entity entity, StandardPatternDataTable patterns){
        this.reInitArchitecture(entity);

        // calculate error
        double errorTraining = 0.0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : patterns) {
            Vector output = neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            for (Numeric real : error) {
                errorTraining += real.doubleValue() * real.doubleValue();
            }
        }
        errorTraining /= patterns.getNumRows() * error.size();
        return errorTraining;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DomainRegistry getDomain() {
        if (!initialised) {
            this.initialise();
        }
        return neuralNetwork.getArchitecture().getDomain();
    }

    /**
     * Gets the datatable builder.
     * @return the datatable builder.
     */
    public DataTableBuilder getDataTableBuilder() {
        return dataTableBuilder;
    }

    /**
     * Sets the datatable builder.
     * @param dataTableBuilder the new datatable builder.
     */
    public void setDataTableBuilder(DataTableBuilder dataTableBuilder) {
        this.dataTableBuilder = dataTableBuilder;
    }

    /**
     * Gets the source URL of the the datatable builder.
     * @return the source URL of the the datatable builder.
     */
    public String getSourceURL() {
        return dataTableBuilder.getSourceURL();
    }

    /**
     * Sets the source URL of the the datatable builder.
     * @param sourceURL the new source URL of the the datatable builder.
     */
    public void setSourceURL(String sourceURL) {
        dataTableBuilder.setSourceURL(sourceURL);
    }

    /**
     * Gets the change frequency value.
     * @return the change frequency value.
     */
    public int getChangeFrequency() {
        return changeFrequency;
    }

    /**
     * Sets the change frequency value.
     * @param changeFrequency the change frequency value.
     */
    public void setChangeFrequency(int changeFrequency) {
        this.changeFrequency = changeFrequency;
    }

    /**
     * Gets the sliding window step size.
     * @return the sliding window step size.
     */
    public int getStepSize() {
        return stepSize;
    }

    /**
     * Sets the sliding window step size.
     * @param stepSize the sliding window step size.
     */
    public void setStepSize(int stepSize) {
        this.stepSize = stepSize;
    }

    /**
     * Gets the sliding window size.
     * @return the sliding window size.
     */
    public int getWindowSize() {
        return windowSize;
    }

    /**
     * Sets the sliding window size.
     * @param windowSize the sliding window size.
     */
    public void setWindowSize(int windowSize) {
        this.windowSize = windowSize;
    }

    /**
     * Sets the classificationSensitivityThreshold
     * @param classificationSensitivityThreshold The classification sensitivity threshold
     */
    public void setClassificationSensitivityThreshold(double classificationSensitivityThreshold){
        this.classificationSensitivityThreshold = classificationSensitivityThreshold;
    }

    /**
     * Gets the classificationSensitivityThreshold
     * @return classification sensitivity threshold
     */
    public double getClassificationSensitivityThreshold(){
        return this.classificationSensitivityThreshold;
    }
}
