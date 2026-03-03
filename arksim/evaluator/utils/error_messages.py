# SPDX-License-Identifier: Apache-2.0
SKIPPED_GOOD_PERFORMANCE_REASON = (
    "All metrics passed threshold, qualitative evaluation skipped"
)

# Evaluation Failure Error Messages
EVALUATION_RUN_FAILED_REASON = "All quantitative metric evaluations encountered errors, so the quantitative evaluation run did not complete."

# Conversation Data Error Messages
NO_CONVERSATION_DATA_WARNING = (
    "Warning: No conversation data found for entry {entry_id}"
)

# Metric Computation Error Messages
METRIC_COMPUTATION_FAILED = "Metric computation failed: {metric_name} - {error_details}"

# File I/O Error Messages
FILE_SAVE_ERROR = "Failed to save file: {file_path} - {error_details}"
FILE_LOAD_ERROR = "Failed to load file: {file_path} - {error_details}"

# API Error Messages
API_KEY_MISSING_ERROR = (
    "The api_key client option must be set either by passing api_key to the client "
    "or by setting the OPENAI_API_KEY environment variable"
)

API_REQUEST_FAILED = "API request failed: {error_details}"

# Validation Error Messages
INVALID_CONVERSATION_FORMAT = "Invalid conversation format: {error_details}"
MISSING_REQUIRED_FIELD = "Missing required field: {field_name}"
INVALID_METRIC_VALUE = "Invalid metric value: {value} for metric {metric_name}"

# Test Error Messages
TEST_DATA_LOAD_FAILED = "Failed to load test data from: {file_path}"
TEST_EVALUATION_FAILED = "Test evaluation failed: {error_details}"
TEST_VALIDATION_FAILED = "Test validation failed: {validation_error}"

# Display Error Messages
DISPLAY_ERROR_GENERAL = "Error displaying results: {error_details}"
DISPLAY_METRICS_ERROR = "Error displaying metrics: {error_details}"
DISPLAY_CONVERSATION_ERROR = "Error displaying conversation: {conversation_id}"
