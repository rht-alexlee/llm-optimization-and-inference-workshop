import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def create_ttft_plot(benchmark_df: pd.DataFrame):
    """
    Generates and displays a line plot of Median Time to First Token (TTFT)
    vs. Poisson Rate from a given DataFrame.

    Args:
        benchmark_df (pd.DataFrame): A DataFrame that must include 'Benchmark'
                                     and 'TTFT_median_ms' columns. The 'Benchmark'
                                     column should be in the format 'name@rate'.
    """
    # --- Data Processing ---

    # Work with a copy to avoid modifying the original DataFrame
    df = benchmark_df.copy()

    # Extract the numeric rate from the 'Benchmark' column for a clean x-axis.
    # This creates a new column called 'Poisson Rate'.
    try:
        # Check if the 'Benchmark' column exists in the DataFrame's columns or index
        if 'Benchmark' in df.columns:
            df['Poisson Rate'] = df['Benchmark'].str.split('@').str[1].astype(float)
        elif df.index.name == 'Benchmark':
             df['Poisson Rate'] = df.index.str.split('@').str[1].astype(float)
        else:
            print("Error: 'Benchmark' column or index not found in the DataFrame.")
            return

    except (IndexError, AttributeError) as e:
        print(f"Error parsing 'Benchmark' column. Ensure it is in 'name@rate' format. Details: {e}")
        return


    # --- Graph Creation ---

    # Set the visual style and size for the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create the line plot
    sns.lineplot(
        x='Poisson Rate',
        y='TTFT_median_ms',
        data=df,
        marker='o', # Add markers to data points for better visibility
        linewidth=2.5
    )

    # Set the title and labels for clarity
    plt.title('Median Time to First Token (TTFT) vs. Poisson Rate', fontsize=16, weight='bold')
    plt.xlabel('Requests per Second (Poisson Rate)', fontsize=12)
    plt.ylabel('Median TTFT (ms)', fontsize=12)

    # Adjust y-axis to start from zero and give space for the highest point
    plt.ylim(0, df['TTFT_median_ms'].max() * 1.15)
    plt.xlim(left=0)

    # Ensure the layout is clean and nothing is cut off
    plt.tight_layout()

    # Display the plot
    plt.show()


# --- Example Usage ---
# This block will only run when the script is executed directly.
# If you import this script as a module elsewhere, this part won't run.
if __name__ == '__main__':
    # Data from your benchmark stats
    data_string = """
Benchmark,TTFT_median_ms
poisson@1.00,355
poisson@4.00,1631
poisson@8.00,2311.2
poisson@16.00,2353
"""
    # First, create the DataFrame from the string data
    source_df = pd.read_csv(io.StringIO(data_string))

    # Now, call the function with the DataFrame
    create_ttft_plot(source_df)