#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <string>
#include <queue>

// Define output paths as constants
const std::string OUTPUT_PATH_SUMMARY = "/Users/pratikpatil/Desktop/Winter 25/risk_metrics.txt";
const std::string OUTPUT_PATH_CSV = "/Users/pratikpatil/Desktop/Winter 25/simulation_results.csv";

// Structure for a limit order
struct Order {
    double price;
    double volume;
    bool isBuy; // true = buy, false = sell
    size_t stepPlaced; // Time step when order was placed (using size_t to avoid warnings)
    enum Status { PENDING, FILLED, CANCELED } status;
};

// Structure for an order book level with comparison operators
struct OrderBookLevel {
    double price;
    double volume;

    // Default comparison for ascending order (needed for std::sort)
    bool operator<(const OrderBookLevel& other) const { return price < other.price; }

    // Comparison for descending order
    bool operator>(const OrderBookLevel& other) const { return price > other.price; }
};

// Class to manage a dynamic limit order book
class OrderBook {
private:
    std::vector<OrderBookLevel> bids; // Sorted descending (highest first)
    std::vector<OrderBookLevel> asks; // Sorted ascending (lowest first)

public:
    void update(double midPrice, int newBuys, int newSells, int cancels,
                std::default_random_engine& gen, std::uniform_real_distribution<double>& volumeDist) {
        // Apply cancellations
        for (int i = 0; i < cancels && (!bids.empty() || !asks.empty()); ++i) {
            if (gen() % 2 == 0 && !bids.empty()) {
                bids.erase(bids.begin() + (gen() % bids.size()));
            } else if (!asks.empty()) {
                asks.erase(asks.begin() + (gen() % asks.size()));
            }
        }

        // Add new buy orders (descending order)
        std::uniform_real_distribution<double> priceDist(midPrice * 0.98, midPrice);
        for (int i = 0; i < newBuys; ++i) {
            bids.push_back({priceDist(gen), volumeDist(gen)});
        }
        // Sort bids in descending order using std::greater
        std::sort(bids.begin(), bids.end(), std::greater<OrderBookLevel>());

        // Add new sell orders (ascending order)
        priceDist = std::uniform_real_distribution<double>(midPrice, midPrice * 1.02);
        for (int i = 0; i < newSells; ++i) {
            asks.push_back({priceDist(gen), volumeDist(gen)});
        }
        // Sort asks in ascending order (default operator<)
        std::sort(asks.begin(), asks.end());
    }

    bool matchOrder(Order& order, double& fillPrice, double& fillVolume) {
        if (order.status != Order::PENDING) return false;

        if (order.isBuy && !asks.empty() && order.price >= asks[0].price) {
            fillPrice = asks[0].price;
            fillVolume = std::min(order.volume, asks[0].volume);
            asks[0].volume -= fillVolume;
            if (asks[0].volume <= 0) asks.erase(asks.begin());
            order.volume -= fillVolume;
            if (order.volume <= 0) order.status = Order::FILLED;
            return true;
        } else if (!order.isBuy && !bids.empty() && order.price <= bids[0].price) {
            fillPrice = bids[0].price;
            fillVolume = std::min(order.volume, bids[0].volume);
            bids[0].volume -= fillVolume;
            if (bids[0].volume <= 0) bids.erase(bids.begin());
            order.volume -= fillVolume;
            if (order.volume <= 0) order.status = Order::FILLED;
            return true;
        }
        return false;
    }

    double getBestBid() const { return bids.empty() ? 0 : bids[0].price; }
    double getBestAsk() const { return asks.empty() ? 0 : asks[0].price; }
    void clear() { bids.clear(); asks.clear(); }
};

// Class to simulate price paths with dynamic LOB
class StochasticModel {
private:
    std::default_random_engine generator;
    std::normal_distribution<double> normalDist{0.0, 1.0};
    std::uniform_real_distribution<double> volumeDist{1.0, 10.0};
    std::poisson_distribution<int> buyArrivalDist{2.0};  // 2 buy orders per step on average
    std::poisson_distribution<int> sellArrivalDist{2.0}; // 2 sell orders per step on average
    std::poisson_distribution<int> cancelDist{1.0};      // 1 cancellation per step on average

public:
    StochasticModel() : generator(std::chrono::system_clock::now().time_since_epoch().count()) {}

    std::vector<double> simulatePath(double initialPrice, double T, size_t steps, double mu, double sigma, OrderBook& book) {
        std::vector<double> path(steps + 1);
        path[0] = initialPrice;
        double dt = T / static_cast<double>(steps);

        // Initial LOB setup
        book.update(initialPrice, buyArrivalDist(generator), sellArrivalDist(generator), 0, generator, volumeDist);

        for (size_t i = 1; i <= steps; ++i) {
            double Z = normalDist(generator);
            path[i] = path[i-1] + mu * path[i-1] * dt + sigma * path[i-1] * std::sqrt(dt) * Z;
            book.update(path[i], buyArrivalDist(generator), sellArrivalDist(generator), cancelDist(generator), generator, volumeDist);
        }
        return path;
    }
};

// Class to perform Bayesian inference (unchanged)
class BayesianEstimator {
private:
    std::default_random_engine generator;
    double priorMuMean, priorMuSd, priorSigmaMean, priorSigmaSd;
    double proposalSdMu, proposalSdSigma;
    int numIterations, burnIn;

    std::vector<double> computeLogReturns(const std::vector<double>& prices) {
        std::vector<double> logReturns;
        for (size_t i = 1; i < prices.size(); ++i) {
            logReturns.push_back(std::log(prices[i] / prices[i-1]));
        }
        return logReturns;
    }

    double logLikelihood(const std::vector<double>& logReturns, double mu, double sigma, double dt) {
        if (sigma <= 0) return -std::numeric_limits<double>::infinity();
        double mean = (mu - 0.5 * sigma * sigma) * dt;
        double variance = sigma * sigma * dt;
        double logL = 0.0;
        for (double r : logReturns) {
            logL += -0.5 * std::log(2 * M_PI * variance) - 0.5 * std::pow(r - mean, 2) / variance;
        }
        return logL;
    }

    double logPriorMu(double mu) {
        return -0.5 * std::pow((mu - priorMuMean) / priorMuSd, 2);
    }

    double logPriorSigma(double sigma) {
        if (sigma <= 0) return -std::numeric_limits<double>::infinity();
        return -0.5 * std::pow((sigma - priorSigmaMean) / priorSigmaSd, 2);
    }

public:
    BayesianEstimator(double priorMuMean, double priorMuSd, double priorSigmaMean, double priorSigmaSd,
                      double proposalSdMu, double proposalSdSigma, int numIterations, int burnIn)
        : priorMuMean(priorMuMean), priorMuSd(priorMuSd), priorSigmaMean(priorSigmaMean), priorSigmaSd(priorSigmaSd),
          proposalSdMu(proposalSdMu), proposalSdSigma(proposalSdSigma), numIterations(numIterations), burnIn(burnIn) {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    std::vector<std::pair<double, double>> estimatePosterior(const std::vector<double>& prices) {
        std::vector<double> logReturns = computeLogReturns(prices);
        double dt = 1.0 / 252.0;

        double sumR = 0.0, sumR2 = 0.0;
        size_t n = logReturns.size();
        for (double r : logReturns) {
            sumR += r;
            sumR2 += r * r;
        }
        double meanR = sumR / n;
        double varR = (sumR2 / n) - (meanR * meanR);
        double sigma_init = std::sqrt(varR / dt);
        double mu_init = (meanR / dt) + 0.5 * sigma_init * sigma_init;

        double mu = mu_init;
        double sigma = sigma_init;
        std::vector<std::pair<double, double>> samples;

        std::normal_distribution<double> proposalMu(0.0, proposalSdMu);
        std::normal_distribution<double> proposalSigma(0.0, proposalSdSigma);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        for (int iter = 0; iter < numIterations; ++iter) {
            double mu_prop = mu + proposalMu(generator);
            double sigma_prop = sigma + proposalSigma(generator);

            double logPostCurrent = logLikelihood(logReturns, mu, sigma, dt) + logPriorMu(mu) + logPriorSigma(sigma);
            double logPostProp = logLikelihood(logReturns, mu_prop, sigma_prop, dt) + logPriorMu(mu_prop) + logPriorSigma(sigma_prop);

            double alpha = std::min(1.0, std::exp(logPostProp - logPostCurrent));
            if (uniform(generator) < alpha) {
                mu = mu_prop;
                sigma = sigma_prop;
            }

            if (iter >= burnIn) {
                samples.push_back({mu, sigma});
            }
        }
        return samples;
    }
};

// Class to apply a trading strategy with true limit orders
class TradingStrategy {
public:
    struct TradeResult {
        double PnL;
        int tradeCount;
        double firstBuyPrice;
        double lastSellPrice;
        bool holdingAtEnd;
        double finalBid;
        double finalAsk;
        size_t pendingOrders; // Using size_t to avoid warnings
    };

    TradeResult apply(const std::vector<double>& path, OrderBook& book, size_t steps) {
        double PnL = 0.0;
        bool holding = false;
        double buyPrice = 0.0;
        int tradeCount = 0;
        double firstBuyPrice = 0.0;
        double lastSellPrice = 0.0;
        std::vector<Order> activeOrders;

        for (size_t t = 1; t <= steps; ++t) {
            double midPrice = path[t];

            // Check and match existing orders
            for (auto it = activeOrders.begin(); it != activeOrders.end();) {
                if (it->status == Order::PENDING) {
                    double fillPrice, fillVolume;
                    if (book.matchOrder(*it, fillPrice, fillVolume)) {
                        if (it->isBuy && it->status == Order::FILLED) {
                            holding = true;
                            buyPrice = fillPrice;
                            if (tradeCount == 0) firstBuyPrice = buyPrice;
                            tradeCount++;
                        } else if (!it->isBuy && it->status == Order::FILLED) {
                            PnL += fillPrice - buyPrice;
                            lastSellPrice = fillPrice;
                            holding = false;
                        }
                    }
                    // Cancel orders older than 10 steps
                    if (t - it->stepPlaced > 10) {
                        it->status = Order::CANCELED;
                    }
                }
                if (it->status != Order::PENDING) {
                    it = activeOrders.erase(it);
                } else {
                    ++it;
                }
            }

            // Place new limit orders based on strategy
            if (!holding && midPrice < path[t-1] * 0.98) { // Bid at 98% of previous price
                activeOrders.push_back({path[t-1] * 0.98, 1.0, true, t, Order::PENDING});
            } else if (holding && midPrice > buyPrice * 1.05) { // Ask at 5% above buy price
                activeOrders.push_back({buyPrice * 1.05, 1.0, false, t, Order::PENDING});
            }
        }

        // Close out holding at end using best bid
        if (holding) {
            double finalBid = book.getBestBid();
            if (finalBid > 0) {
                PnL += finalBid - buyPrice;
                lastSellPrice = finalBid;
            }
        }

        size_t pendingOrdersCount = 0;
        for (const auto& order : activeOrders) {
            if (order.status == Order::PENDING) ++pendingOrdersCount;
        }
        return {PnL, tradeCount, firstBuyPrice, lastSellPrice, holding, book.getBestBid(), book.getBestAsk(), pendingOrdersCount};
    }
};

// Class to run Monte Carlo simulations with LOB
class MonteCarloSimulator {
private:
    StochasticModel model;
    TradingStrategy strategy;

public:
    void run(size_t numSimulations, double initialPrice, double T, size_t steps,
             const std::vector<std::pair<double, double>>& posteriorSamples) {
        std::vector<double> PnLs(numSimulations);
        std::vector<double> mus(numSimulations);
        std::vector<double> sigmas(numSimulations);
        std::vector<double> finalPrices(numSimulations);
        std::vector<int> tradeCounts(numSimulations);
        std::vector<double> firstBuyPrices(numSimulations);
        std::vector<double> lastSellPrices(numSimulations);
        std::vector<int> tradeFlags(numSimulations);
        std::vector<double> finalBids(numSimulations);
        std::vector<double> finalAsks(numSimulations);
        std::vector<size_t> pendingOrders(numSimulations);

        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> distribution(0, posteriorSamples.size() - 1);

        double sum = 0.0, sumSq = 0.0;
        for (size_t sim = 0; sim < numSimulations; ++sim) {
            size_t idx = distribution(generator);
            double mu = posteriorSamples[idx].first;
            double sigma = posteriorSamples[idx].second;
            OrderBook book;
            std::vector<double> path = model.simulatePath(initialPrice, T, steps, mu, sigma, book);
            TradingStrategy::TradeResult result = strategy.apply(path, book, steps);

            PnLs[sim] = result.PnL;
            mus[sim] = mu;
            sigmas[sim] = sigma;
            finalPrices[sim] = path.back();
            tradeCounts[sim] = result.tradeCount;
            firstBuyPrices[sim] = result.firstBuyPrice;
            lastSellPrices[sim] = result.lastSellPrice;
            tradeFlags[sim] = result.holdingAtEnd ? 1 : 0;
            finalBids[sim] = result.finalBid;
            finalAsks[sim] = result.finalAsk;
            pendingOrders[sim] = result.pendingOrders;

            sum += result.PnL;
            sumSq += result.PnL * result.PnL;
        }

        // Compute summary statistics
        double meanPnL = sum / numSimulations;
        double variance = (sumSq / numSimulations) - (meanPnL * meanPnL);
        double stdDev = std::sqrt(variance);

        std::sort(PnLs.begin(), PnLs.end());
        size_t idx5 = static_cast<size_t>(0.05 * numSimulations);
        double VaR5 = PnLs[idx5];
        double sumCVaR = 0.0;
        for (size_t i = 0; i <= idx5; ++i) {
            sumCVaR += PnLs[i];
        }
        double CVaR5 = sumCVaR / (idx5 + 1);

        // Write summary to risk_metrics.txt
        std::ofstream summaryFile(OUTPUT_PATH_SUMMARY);
        if (!summaryFile.is_open()) {
            std::cerr << "Error: Could not open summary file for writing: " << OUTPUT_PATH_SUMMARY << "\n";
            return;
        }
        summaryFile << "Mean P&L: " << meanPnL << "\n";
        summaryFile << "Standard Deviation of P&L: " << stdDev << "\n";
        summaryFile << "5% VaR: " << VaR5 << "\n";
        summaryFile << "5% CVaR: " << CVaR5 << "\n";
        summaryFile.close();

        // Write all data to CSV
        std::ofstream csvFile(OUTPUT_PATH_CSV);
        if (!csvFile.is_open()) {
            std::cerr << "Error: Could not open CSV file for writing: " << OUTPUT_PATH_CSV << "\n";
            return;
        }
        // Header
        csvFile << "Simulation,P&L,Mu,Sigma,FinalPrice,TradeCount,FirstBuyPrice,LastSellPrice,TradeFlag,FinalBid,FinalAsk,PendingOrders\n";
        // All simulation data
        for (size_t i = 0; i < numSimulations; ++i) {
            csvFile << i + 1 << "," << PnLs[i] << "," << mus[i] << "," << sigmas[i] << ","
                    << finalPrices[i] << "," << tradeCounts[i] << ","
                    << (firstBuyPrices[i] == 0 ? "NA" : std::to_string(firstBuyPrices[i])) << ","
                    << (lastSellPrices[i] == 0 ? "NA" : std::to_string(lastSellPrices[i])) << ","
                    << tradeFlags[i] << "," << finalBids[i] << "," << finalAsks[i] << "," << pendingOrders[i] << "\n";
        }
        // Append summary stats
        csvFile << "\nSummary Statistics\n";
        csvFile << "Mean P&L," << meanPnL << "\n";
        csvFile << "Standard Deviation," << stdDev << "\n";
        csvFile << "5% VaR," << VaR5 << "\n";
        csvFile << "5% CVaR," << CVaR5 << "\n";
        csvFile.close();
    }
};

// Main function to drive the simulation
int main() {
    // Parameters
    double true_mu = 0.1;       // True drift (10% annual return)
    double true_sigma = 0.2;    // True volatility (20% annual volatility)
    double initialPrice = 100.0; // Starting price
    double T = 1.0;             // 1 year
    size_t steps = 252;         // 252 trading days (using size_t to avoid warnings)

    // Step 1: Generate synthetic historical prices
    StochasticModel model;
    OrderBook dummyBook;
    std::vector<double> historicalPrices = model.simulatePath(initialPrice, T, steps, true_mu, true_sigma, dummyBook);

    // Step 2: Estimate posterior distribution
    double priorMuMean = 0.0, priorMuSd = 1.0;
    double priorSigmaMean = 0.2, priorSigmaSd = 0.1;
    double proposalSdMu = 0.01, proposalSdSigma = 0.01;
    int numIterations = 11000, burnIn = 1000;

    BayesianEstimator estimator(priorMuMean, priorMuSd, priorSigmaMean, priorSigmaSd,
                               proposalSdMu, proposalSdSigma, numIterations, burnIn);
    std::vector<std::pair<double, double>> posteriorSamples = estimator.estimatePosterior(historicalPrices);

    // Step 3: Run Monte Carlo simulations
    size_t numSimulations = 1000000; // 1,000,000 simulations (using size_t to avoid warnings)
    double futureInitialPrice = historicalPrices.back();
    MonteCarloSimulator simulator;
    simulator.run(numSimulations, futureInitialPrice, T, steps, posteriorSamples);

    std::cout << "Simulation complete.\n";
    std::cout << "Summary metrics written to '" << OUTPUT_PATH_SUMMARY << "'.\n";
    std::cout << "Full results written to '" << OUTPUT_PATH_CSV << "'.\n";
    return 0;
}
