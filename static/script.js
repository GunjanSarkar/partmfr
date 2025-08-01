/**
 * Part Number Lookup Application
 * Handles search and batch processing of motor part numbers
 */

class PartLookupApp {
    constructor() {
        this.batchData = null;
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.bindEvents();
        this.focusInput();
    }
    
    /**
     * Bind all event listeners
     */
    bindEvents() {
        // Get UI elements
        const searchBtn = document.getElementById('searchBtn');
        const partNumberInput = document.getElementById('partNumber');
        const partDescriptionInput = document.getElementById('partDescription');
        const singleSearchTab = document.getElementById('singleSearchTab');
        const batchProcessingTab = document.getElementById('batchProcessingTab');
        const singleSearchPanel = document.getElementById('singleSearchPanel');
        const batchProcessingPanel = document.getElementById('batchProcessingPanel');
        const fileInput = document.getElementById('fileInput');
        const fileSelectBtn = document.getElementById('fileSelectBtn');
        const clearFileBtn = document.getElementById('clearFileBtn');
        const processBtn = document.getElementById('processBtn');
        const downloadResultsBtn = document.getElementById('downloadResultsBtn');

        // Single search events
        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.handleSearch());
        }
        
        if (partNumberInput) {
            partNumberInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch();
                }
            });
            
            partNumberInput.addEventListener('input', () => {
                this.clearResults();
            });
        }
        
        if (partDescriptionInput) {
            partDescriptionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch();
                }
            });
            
            partDescriptionInput.addEventListener('input', () => {
                this.clearResults();
            });
        }
        
        // Tab switching
        if (singleSearchTab && batchProcessingTab) {
            singleSearchTab.addEventListener('click', () => {
                singleSearchTab.classList.add('active');
                batchProcessingTab.classList.remove('active');
                singleSearchPanel.classList.remove('hidden');
                batchProcessingPanel.classList.add('hidden');
                this.focusInput();
                // Clear batch results when switching away from batch tab
                this.hideBatchResults();
            });
            
            batchProcessingTab.addEventListener('click', () => {
                batchProcessingTab.classList.add('active');
                singleSearchTab.classList.remove('active');
                batchProcessingPanel.classList.remove('hidden');
                singleSearchPanel.classList.add('hidden');
            });
        }
        
        // File input events
        if (fileSelectBtn && fileInput) {
            fileSelectBtn.addEventListener('click', () => {
                fileInput.click();
            });
        }
        
        if (clearFileBtn && fileInput) {
            clearFileBtn.addEventListener('click', () => {
                fileInput.value = '';
                this.updateFileStatus();
            });
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.updateFileStatus();
            });
        }
        
        // Upload button events
        if (processBtn) {
            processBtn.addEventListener('click', () => this.handleFileUpload());
        }
        
        // Download button events
        if (downloadResultsBtn) {
            downloadResultsBtn.addEventListener('click', () => this.downloadBatchResults());
        }
        
        // Attach event handler for details buttons - using event delegation
        document.addEventListener('click', (e) => {
            if (e.target && e.target.classList.contains('details-btn')) {
                const partNumber = e.target.getAttribute('data-part');
                this.toggleDetails(partNumber);
            }
            
            // // Handle download results button
            // if (e.target && e.target.id === 'downloadResultsBtn') {
            //     this.downloadResults();
            // }
        });
    }
    
    /**
     * Toggle details row visibility
     */
    toggleDetails(partNumber) {
        if (!partNumber) return;
        
        const detailsRow = document.getElementById(`details-${partNumber}`);
        if (detailsRow) {
            detailsRow.classList.toggle('hidden');
        }
    }
    
    /**
     * Format file size in readable format
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    /**
     * Set focus to the part number input
     */
    focusInput() {
        const partNumberInput = document.getElementById('partNumber');
        if (partNumberInput) {
            setTimeout(() => {
                partNumberInput.focus();
            }, 100);
        }
    }
    
    /**
     * Clear any existing results
     */
    clearResults() {
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.classList.add('hidden');
        }
        
        // Clear any batch results too
        this.hideBatchResults();
    }
    
    /**
     * Hide batch results
     */
    hideBatchResults() {
        const batchResultsSection = document.getElementById('batchResultsSection');
        const downloadBtn = document.getElementById('downloadResultsBtn');
        
        if (batchResultsSection) {
            batchResultsSection.classList.add('hidden');
        }
        
        if (downloadBtn) {
            downloadBtn.classList.add('hidden');
        }
        
        // Clear stored batch data
        this.batchData = null;
        this.currentFileName = null;
    }
    
    /**
     * Clear any error messages
     */
    clearError() {
        const errorContainer = document.getElementById('errorContainer');
        if (errorContainer) {
            errorContainer.classList.add('hidden');
            errorContainer.textContent = '';
        }
    }
    
    /**
     * Show an error message
     */
    showError(message) {
        const errorContainer = document.getElementById('errorContainer');
        const errorSection = document.getElementById('errorSection');
        const errorMessage = document.getElementById('errorMessage');
        
        if (errorSection && errorMessage) {
            errorSection.classList.remove('hidden');
            errorMessage.textContent = message;
        } else if (errorContainer) {
            errorContainer.classList.remove('hidden');
            errorContainer.textContent = message;
        } else {
            alert(message);
        }
        
        this.hideLoading();
    }
    
    /**
     * Show the loading indicator
     */
    showLoading(message = 'Processing part number...') {
        const loadingIndicator = document.getElementById('loadingIndicator');
        const loadingMessage = document.getElementById('loadingMessage');
        
        if (loadingIndicator) {
            loadingIndicator.classList.remove('hidden');
        }
        
        if (loadingMessage) {
            loadingMessage.textContent = message;
        }
    }
    
    /**
     * Hide the loading indicator
     */
    hideLoading() {
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.classList.add('hidden');
        }
    }
    
    /**
     * Handle the search action
     */
    async handleSearch() {
        const partNumberInput = document.getElementById('partNumber');
        const partDescriptionInput = document.getElementById('partDescription');
        
        if (!partNumberInput || !partNumberInput.value.trim()) {
            this.showError('Please enter a part number');
            return;
        }
        
        const partNumber = partNumberInput.value.trim();
        const partDescription = partDescriptionInput ? partDescriptionInput.value.trim() : '';
        
        this.showLoading();
        this.clearResults();
        this.clearError();
        
        try {
            // Prepare request body
            const requestBody = {
                part_number: partNumber
            };
            
            // Add description if provided
            if (partDescription) {
                requestBody.part_description = partDescription;
            }
            
            // Send request
            const response = await fetch('/api/process-part', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                let errorMessage = `HTTP error: ${response.status} ${response.statusText}`;
                try {
                    const errorText = await response.text();
                    if (errorText) {
                        try {
                            const errorJson = JSON.parse(errorText);
                            if (errorJson.detail) {
                                errorMessage = errorJson.detail;
                            }
                        } catch (e) {
                            errorMessage = errorText;
                        }
                    }
                } catch (e) {
                    // Ignore error parsing
                }
                
                throw new Error(errorMessage);
            }
            
            const data = await response.json();
            console.log("Received data:", data);
            
            this.displayResults(data);
            this.displayExecutionTime(data.execution_time);
            
        } catch (error) {
            this.showError(`Error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    /**
     * Handle file upload for batch processing
     */
    async handleFileUpload() {
        const fileInput = document.getElementById('fileInput');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            this.showError('Please select a file to upload');
            return;
        }
        
        const file = fileInput.files[0];
        
        this.showLoading('Processing batch file...');
        this.clearResults();
        this.clearError();
        this.hideBatchResults();
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/process-file?output_format=json', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                let errorMessage = `HTTP error: ${response.status} ${response.statusText}`;
                try {
                    const errorText = await response.text();
                    if (errorText) {
                        try {
                            const errorJson = JSON.parse(errorText);
                            if (errorJson.detail) {
                                errorMessage = errorJson.detail;
                            }
                        } catch (e) {
                            errorMessage = errorText;
                        }
                    }
                } catch (e) {
                    // Ignore error parsing
                }
                
                throw new Error(errorMessage);
            }
            
            const data = await response.json();
            
            // Store the results for CSV download
            this.batchData = data;
            this.currentFileName = file.name;
            
            this.displayBatchResults(data);
            this.displayBatchExecutionTime(data.execution_time);
            
            // Show the download button now that we have results
            const downloadBtn = document.getElementById('downloadResultsBtn');
            if (downloadBtn) {
                downloadBtn.classList.remove('hidden');
            }
            
        } catch (error) {
            this.showError(`Error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    /**
     * Display the search results
     */
    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const noResultsContainer = document.getElementById('noResultsContainer');
        const manufacturerPartsContainer = document.getElementById('manufacturerPartsContainer');
        const remanufacturerPartsContainer = document.getElementById('remanufacturerPartsContainer');
        
        if (!resultsSection) return;
        
        // Check if we have results
        if (data.status === 'success' && data.filtered_results && data.filtered_results.length > 0) {
            resultsSection.classList.remove('hidden');
            
            // Hide no results message
            if (noResultsContainer) {
                noResultsContainer.classList.add('hidden');
            }
            
            // Display manufacturer parts
            if (manufacturerPartsContainer) {
                let html = '';
                
                html += `<div class="result-block">
                    <h3>Manufacturer Parts</h3>
                    <div class="result-grid">`;
                
                data.filtered_results.forEach(part => {
                    html += `
                    <div class="result-card">
                        <div class="card-header">
                            <div class="part-info">
                                <span class="part-number">${part.PARTNUMBER}</span>
                                <span class="part-class">${part.CLASS}</span>
                            </div>
                            <div class="manufacturer">${part.PARTMFR || 'Unknown'}</div>
                        </div>
                        <div class="card-body">
                            <div class="description">${part.partdesc || 'No description available'}</div>
                            ${part.cocode ? `<div class="scoring-info">
                                <span class="confidence-code">Code: ${part.cocode}</span>
                                <span class="scores">Part: ${part.part_number_score || 0}/5 | Desc: ${part.description_score || 0}/5 | Noise: ${part.noise_detected === 1 ? 'Yes' : 'No'}</span>
                            </div>` : ''}
                        </div>
                        <div class="card-footer">
                            <div class="part-id">ID: ${part.PARTINDEX}</div>
                            <div class="confidence">${part.confidence ? `Confidence: ${Math.round(part.confidence * 100)}%` : ''}</div>
                        </div>
                    </div>
                    `;
                });
                
                html += `</div></div>`;
                manufacturerPartsContainer.innerHTML = html;
                manufacturerPartsContainer.classList.remove('hidden');
            }
            
            // Display remanufacturer parts if available
            if (remanufacturerPartsContainer) {
                if (data.remanufacturer_variants && data.remanufacturer_variants.length > 0) {
                    let html = '';
                    
                    html += `<div class="result-block">
                        <h3>Remanufacturer Variants</h3>
                        <div class="result-grid">`;
                    
                    data.remanufacturer_variants.forEach(part => {
                        html += `
                        <div class="result-card variant">
                            <div class="card-header">
                                <div class="part-info">
                                    <span class="part-number">${part.PARTNUMBER}</span>
                                    <span class="part-class">${part.CLASS}</span>
                                </div>
                                <div class="manufacturer">${part.PARTMFR || 'Unknown'}</div>
                            </div>
                            <div class="card-body">
                                <div class="description">${part.partdesc || 'No description available'}</div>
                                <div class="variant-note">Variant of: ${part.original_part || 'Unknown'}</div>
                                ${part.cocode ? `<div class="scoring-info">
                                    <span class="confidence-code">Code: ${part.cocode}</span>
                                    <span class="scores">Part: ${part.part_number_score || 0}/5 | Desc: ${part.description_score || 0}/5 | Noise: ${part.noise_detected === 1 ? 'Yes' : 'No'}</span>
                                </div>` : ''}
                            </div>
                            <div class="card-footer">
                                <div class="part-id">ID: ${part.PARTINDEX}</div>
                                <div class="similarity">${part.similarity ? `Similarity: ${Math.round(part.similarity * 100)}%` : ''}</div>
                            </div>
                        </div>
                        `;
                    });
                    
                    html += `</div></div>`;
                    remanufacturerPartsContainer.innerHTML = html;
                    remanufacturerPartsContainer.classList.remove('hidden');
                } else {
                    remanufacturerPartsContainer.classList.add('hidden');
                }
            }
        } else {
            // Show no results message
            if (noResultsContainer) {
                noResultsContainer.classList.remove('hidden');
                
                // Add cleaned part info and error reason if available
                let noResultsHtml = '<h3>No Results Found</h3>';
                if (data.cleaned_part) {
                    noResultsHtml += `<p>Cleaned part number: <strong>${data.cleaned_part}</strong></p>`;
                }
                if (data.error_reason) {
                    noResultsHtml += `<p class="error-reason">${data.error_reason}</p>`;
                }
                noResultsContainer.innerHTML = noResultsHtml;
            }
            
            // Hide results containers
            if (manufacturerPartsContainer) {
                manufacturerPartsContainer.classList.add('hidden');
            }
            if (remanufacturerPartsContainer) {
                remanufacturerPartsContainer.classList.add('hidden');
            }
        }
        
        // Show the overall results section
        resultsSection.classList.remove('hidden');
    }
    
    /**
     * Display batch processing results
     */
    displayBatchResults(data) {
        const batchResultsContainer = document.getElementById('batchResultsContainer');
        const batchResultsSection = document.getElementById('batchResultsSection');
        const batchSummary = document.getElementById('batchSummary');
        
        if (!batchResultsContainer || !batchResultsSection || !batchSummary) {
            return;
        }
        
        // Display summary
        batchSummary.innerHTML = `
            <div class="summary-item">
                <span class="label">Total</span>
                <span class="value">${data.total_processed}</span>
            </div>
            <div class="summary-item">
                <span class="label">Successful</span>
                <span class="value">${data.successful}</span>
            </div>
            <div class="summary-item">
                <span class="label">Failed</span>
                <span class="value">${data.failed}</span>
            </div>
        `;
        
        // Create results table
        let html = `
        <table class="batch-results-table">
            <thead>
                <tr>
                    <th>Original Part</th>
                    <th>Description</th>
                    <th>Cleaned Part</th>
                    <th>Status</th>
                    <th>Results Found</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        `;
        
        if (data.results && data.results.length > 0) {
            data.results.forEach(resultItem => {
                const part_number = resultItem.part_number || 'N/A';
                const description = resultItem.description || 'N/A';
                const result = resultItem.result || {};
                const cleaned_part = result.cleaned_part || 'N/A';
                const status = result.status || 'failed';
                const filtered_results = result.filtered_results || [];
                const remanufacturer_variants = result.remanufacturer_variants || [];
                const resultCount = filtered_results.length + remanufacturer_variants.length;
                
                html += `
                <tr class="${status === 'success' ? 'success' : 'error'}">
                    <td class="part-number-cell">${part_number}</td>
                    <td class="description-cell">${description}</td>
                    <td class="cleaned-part-cell">${cleaned_part}</td>
                    <td class="status-cell">
                        <span class="status-badge ${status === 'success' ? 'success' : 'error'}">
                            ${status}
                        </span>
                    </td>
                    <td class="results-count-cell">
                        ${resultCount > 0 ? `<span class="results-count">${resultCount} results</span>` : '<span class="no-results">No results</span>'}
                    </td>
                    <td class="actions-cell">
                        ${resultCount > 0 ? `<button class="details-btn" data-part="${part_number}">View Details</button>` : ''}
                    </td>
                </tr>
                `;
                
                // Add collapsible details section
                if (status === 'success' && resultCount > 0) {
                    html += `
                    <tr class="details-row hidden" id="details-${part_number}">
                        <td colspan="6">
                            <div class="details-content">
                    `;
                    
                    // Add manufacturer parts
                    if (filtered_results && filtered_results.length > 0) {
                        html += `<h4>Manufacturer Parts (${filtered_results.length})</h4>
                        <div class="details-grid">`;
                        
                        filtered_results.forEach(part => {
                            html += `
                            <div class="detail-card">
                                <div class="detail-header">
                                    <span class="part-number">${part.PARTNUMBER}</span>
                                    <span class="class-badge">${part.CLASS}</span>
                                </div>
                                <div class="detail-body">
                                    <p><strong>Manufacturer:</strong> ${part.PARTMFR || 'Unknown'}</p>
                                    <p><strong>Description:</strong> ${part.partdesc || 'No description'}</p>
                                    <p><strong>Part ID:</strong> ${part.PARTINDEX}</p>
                                    ${part.confidence ? `<p><strong>Confidence:</strong> ${Math.round(part.confidence * 100)}%</p>` : ''}
                                    ${part.cocode ? `<div class="scoring-info">
                                        <span class="confidence-code">Code: ${part.cocode}</span>
                                        <span class="scores">Part: ${part.part_number_score || 0}/5 | Desc: ${part.description_score || 0}/5 | Noise: ${part.noise_detected === 1 ? 'Yes' : 'No'}</span>
                                    </div>` : ''}
                                </div>
                            </div>
                            `;
                        });
                        
                        html += `</div>`;
                    }
                    
                    // Add remanufacturer parts
                    if (remanufacturer_variants && remanufacturer_variants.length > 0) {
                        html += `<h4>Remanufacturer Variants (${remanufacturer_variants.length})</h4>
                        <div class="details-grid">`;
                        
                        remanufacturer_variants.forEach(part => {
                            html += `
                            <div class="detail-card variant">
                                <div class="detail-header">
                                    <span class="part-number">${part.PARTNUMBER}</span>
                                    <span class="class-badge">${part.CLASS}</span>
                                </div>
                                <div class="detail-body">
                                    <p><strong>Manufacturer:</strong> ${part.PARTMFR || 'Unknown'}</p>
                                    <p><strong>Description:</strong> ${part.partdesc || 'No description'}</p>
                                    <p><strong>Part ID:</strong> ${part.PARTINDEX}</p>
                                    <p><strong>Original Part:</strong> ${part.original_part || 'Unknown'}</p>
                                    ${part.similarity ? `<p><strong>Similarity:</strong> ${Math.round(part.similarity * 100)}%</p>` : ''}
                                    ${part.cocode ? `<div class="scoring-info">
                                        <span class="confidence-code">Code: ${part.cocode}</span>
                                        <span class="scores">Part: ${part.part_number_score || 0}/5 | Desc: ${part.description_score || 0}/5 | Noise: ${part.noise_detected === 1 ? 'Yes' : 'No'}</span>
                                    </div>` : ''}
                                </div>
                            </div>
                            `;
                        });
                        
                        html += `</div>`;
                    }
                    
                    html += `
                            </div>
                        </td>
                    </tr>
                    `;
                }
            });
        } else {
            html += `
            <tr>
                <td colspan="6" class="no-results">No results available</td>
            </tr>
            `;
        }
        
        html += `
            </tbody>
        </table>
        `;
        
        // We don't need to add a download button since it already exists in the HTML
        batchResultsContainer.innerHTML = html;
        batchResultsSection.classList.remove('hidden');
    }
    
    /**
     * Download batch processing results as CSV
     */
    downloadResults() {
        if (!this.batchData || !this.batchData.results || this.batchData.results.length === 0) {
            this.showError('No results available to download');
            return;
        }
        
        const fileInput = document.getElementById('fileInput');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            this.showError('Original file information not found');
            return;
        }
        
        const file = fileInput.files[0];
        
        // Show loading indicator
        this.showLoading('Preparing CSV download...');
        
        // Create form data with the file
        const formData = new FormData();
        formData.append('file', file);
        
        // Direct browser to download URL
        fetch('/api/process-file?output_format=csv', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to generate CSV file (Status: ${response.status})`);
            }
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `${file.name.split('.')[0]}_results.csv`;
            
            // Trigger download
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
            window.URL.revokeObjectURL(url);
            a.remove();
            this.hideLoading();
        })
        .catch(error => {
            this.showError(`Error downloading CSV: ${error.message}`);
            this.hideLoading();
        });
    }
    
    /**
     * Download batch processing results as CSV using stored data
     */
    async downloadBatchResults() {
        if (!this.batchData || !this.batchData.results || this.batchData.results.length === 0) {
            this.showError('No batch results available to download. Please process a file first.');
            return;
        }

        if (!this.currentFileName) {
            this.showError('Original filename not available.');
            return;
        }

        try {
            // Show loading for download
            this.showLoading('Preparing CSV download...');

            // Generate CSV content directly from stored results
            const csvContent = this.generateCSVFromResults(this.batchData.results);
            
            // Create a blob from the CSV data and trigger download
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = window.URL.createObjectURL(blob);
            
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = this.currentFileName.replace(/\.[^/.]+$/, '') + '_results.csv';
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            // Clean up the URL
            window.URL.revokeObjectURL(url);

        } catch (error) {
            this.showError(`Error downloading CSV: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Create FormData from stored results (fallback method)
     */
    /**
     * Generate CSV content from stored results
     */
    generateCSVFromResults(results) {
        // Define CSV headers in the exact order required
        const headers = [
            'input_part_number',
            'input_description', 
            'cleaned_part_number',
            'part_number_score',
            'description_score',
            'noise_detected',
            'cocode',
            'PARTINDEX',
            'CLASS',
            'PARTMFR',
            'PARTNUMBER', 
            'SPARTNUMBER',
            'partdesc',
            'classification',
            'execution_time_ms',
            'success',
            'error_message'
        ];

        // Create CSV content starting with headers
        let csvContent = headers.join(',') + '\n';

        // Add each result row
        results.forEach(result => {
            const row = [
                this.escapeCSVField(result.input_part_number || ''),
                this.escapeCSVField(result.input_description || ''),
                this.escapeCSVField(result.cleaned_part_number || ''),
                result.part_number_score || '',
                result.description_score || '',
                result.noise_detected || 0,
                result.cocode || '',
                result.PARTINDEX || '',
                this.escapeCSVField(result.CLASS || ''),
                this.escapeCSVField(result.PARTMFR || ''),
                this.escapeCSVField(result.PARTNUMBER || ''),
                this.escapeCSVField(result.SPARTNUMBER || ''),
                this.escapeCSVField(result.partdesc || ''),
                this.escapeCSVField(result.classification || ''),
                result.execution_time_ms || '',
                result.success || '',
                this.escapeCSVField(result.error_message || '')
            ];
            csvContent += row.join(',') + '\n';
        });

        return csvContent;
    }

    /**
     * Escape CSV field to handle commas, quotes, and newlines
     */
    escapeCSVField(field) {
        if (field === null || field === undefined) {
            return '';
        }
        
        const stringField = String(field);
        
        // If field contains comma, quote, or newline, wrap in quotes and escape internal quotes
        if (stringField.includes(',') || stringField.includes('"') || stringField.includes('\n')) {
            return '"' + stringField.replace(/"/g, '""') + '"';
        }
        
        return stringField;
    }

    createFormDataFromResults() {
        // This is a fallback - ideally we'd convert stored JSON results to CSV directly
        // For now, we'll need to re-upload the file, but we could enhance this later
        const fileInput = document.getElementById('fileInput');
        if (fileInput && fileInput.files && fileInput.files[0]) {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            return formData;
        }
        throw new Error('Original file not available for CSV generation');
    }

    /**
     * Display execution time
     */
    displayExecutionTime(seconds) {
        const executionTimeContainer = document.getElementById('executionTimeContainer');
        const executionTimeElement = document.getElementById('executionTime');
        
        if (executionTimeContainer && executionTimeElement) {
            executionTimeElement.textContent = seconds.toFixed(3);
            executionTimeContainer.classList.remove('hidden');
        }
    }
    
    /**
     * Display batch execution time
     */
    displayBatchExecutionTime(seconds) {
        const batchExecutionTimeContainer = document.getElementById('batchExecutionTimeContainer');
        const batchExecutionTimeElement = document.getElementById('batchExecutionTime');
        
        if (batchExecutionTimeContainer && batchExecutionTimeElement) {
            batchExecutionTimeElement.textContent = seconds.toFixed(3);
            batchExecutionTimeContainer.classList.remove('hidden');
        }
    }
    
    /**
     * Update file status display
     */
    updateFileStatus() {
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const clearFileBtn = document.getElementById('clearFileBtn');
        const uploadBtn = document.getElementById('uploadBtn');
        const processBtn = document.getElementById('processBtn');
        const fileSelectBtn = document.getElementById('fileSelectBtn');
        
        if (!fileInput || !fileInfo) return;
        
        const file = fileInput.files[0];
        
        if (file) {
            // Check if file type is supported
            const fileExtension = file.name.split('.').pop().toLowerCase();
            if (['csv', 'xlsx', 'xls'].indexOf(fileExtension) === -1) {
                fileInfo.innerHTML = `<span class="error">Invalid file type. Please select a CSV or Excel file.</span>`;
                if (processBtn) processBtn.disabled = true;
                if (clearFileBtn) clearFileBtn.classList.add('hidden');
                if (fileSelectBtn) {
                    fileSelectBtn.innerHTML = '<span class="file-icon">📁</span>Choose File';
                    fileSelectBtn.classList.remove('file-selected');
                }
                return;
            }
            
            // Display file info
            fileInfo.innerHTML = `<span class="file-name">${file.name}</span> <span class="file-size">(${this.formatFileSize(file.size)})</span>`;
            if (processBtn) processBtn.disabled = false;
            if (clearFileBtn) clearFileBtn.classList.remove('hidden');
            if (fileSelectBtn) {
                fileSelectBtn.innerHTML = '<span class="file-icon">📄</span>Change File';
                fileSelectBtn.classList.add('file-selected');
            }
            
            // Clear previous batch results when new file is selected
            this.hideBatchResults();
        } else {
            fileInfo.textContent = 'No file selected';
            if (processBtn) processBtn.disabled = true;
            if (clearFileBtn) clearFileBtn.classList.add('hidden');
            if (fileSelectBtn) {
                fileSelectBtn.innerHTML = '<span class="file-icon">📁</span>Choose File';
                fileSelectBtn.classList.remove('file-selected');
            }
            
            // Hide batch results if no file selected
            this.hideBatchResults();
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PartLookupApp();
});
