#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <ctime>
#include <uuid/uuid.h>
#include <set>
#include <cstring>
#include <numeric>

// enum for trade type
enum class TradeType { Buy, Sell };

// trade order
struct TradeOrder {
    std::string orderId;
    double price;
    int quantity;
    uint64_t timestamp;
    TradeType type;

    TradeOrder() : price(0.0), quantity(0), timestamp(0), type(TradeType::Buy) {}

    TradeOrder(const std::string& id, double p, int q, uint64_t ts, TradeType t)
        : orderId(id), price(p), quantity(q), timestamp(ts), type(t) {}

    bool operator<(const TradeOrder& other) const {
        if (quantity != other.quantity)
            return quantity < other.quantity;
        return price < other.price;
    }

    // Serialization: Convert TradeOrder to a byte buffer
    std::vector<char> serialize() const {
        std::vector<char> buffer;
        size_t orderIdSize = orderId.size();
        buffer.resize(sizeof(orderIdSize) + orderIdSize + sizeof(price) +
                      sizeof(quantity) + sizeof(timestamp) + sizeof(type));
        char* ptr = buffer.data();
        memcpy(ptr, &orderIdSize, sizeof(orderIdSize));
        ptr += sizeof(orderIdSize);
        memcpy(ptr, orderId.c_str(), orderIdSize);
        ptr += orderIdSize;
        memcpy(ptr, &price, sizeof(price));
        ptr += sizeof(price);
        memcpy(ptr, &quantity, sizeof(quantity));
        ptr += sizeof(quantity);
        memcpy(ptr, &timestamp, sizeof(timestamp));
        ptr += sizeof(timestamp);
        memcpy(ptr, &type, sizeof(type));
        return buffer;
    }

    // Deserialization: Convert a byte buffer to a TradeOrder
    void deserialize(const std::vector<char>& buffer) {
        const char* ptr = buffer.data();
        size_t orderIdSize;
        memcpy(&orderIdSize, ptr, sizeof(orderIdSize));
        ptr += sizeof(orderIdSize);
        orderId.assign(ptr, orderIdSize);
        ptr += orderIdSize;
        memcpy(&price, ptr, sizeof(price));
        ptr += sizeof(price);
        memcpy(&quantity, ptr, sizeof(quantity));
        ptr += sizeof(quantity);
        memcpy(&timestamp, ptr, sizeof(timestamp));
        ptr += sizeof(timestamp);
        memcpy(&type, ptr, sizeof(type));
    }
};

// utility functions
std::string generateUUID() {
    uuid_t uuid;
    uuid_generate(uuid);
    char uuidStr[37];
    uuid_unparse(uuid, uuidStr);
    return std::string(uuidStr);
}

std::string formatTimestamp(uint64_t nanoseconds) {
    time_t seconds = nanoseconds / 1'000'000'000;
    struct tm* timeInfo = gmtime(&seconds);
    char buffer[30];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeInfo);
    return std::string(buffer);
}

std::vector<TradeOrder> generateTradeOrders(size_t numberOfOrders) {
    std::vector<TradeOrder> orders;
    orders.reserve(numberOfOrders);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> priceDist(0.000001, 1000.000000);
    std::uniform_int_distribution<int> quantityDist(1, 10000);
    std::uniform_int_distribution<int> typeDist(0, 1);

    uint64_t startEpoch = 1704067200000000000ULL;
    uint64_t endEpoch = 1735689599000000000ULL;
    std::uniform_int_distribution<uint64_t> timestampDist(startEpoch, endEpoch);

    for (size_t i = 0; i < numberOfOrders; ++i) {
        orders.emplace_back(
            generateUUID(),
            std::round(priceDist(gen) * 1'000'000) / 1'000'000,
            quantityDist(gen),
            timestampDist(gen),
            static_cast<TradeType>(typeDist(gen))
        );
    }
    return orders;
}

// output functions
void outputTradeOrders(const std::vector<TradeOrder>& orders, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for output." << std::endl;
        return;
    }

    outFile << std::left << std::setw(40) << "OrderID"
            << std::setw(15) << "Price"
            << std::setw(10) << "Quantity"
            << std::setw(25) << "Timestamp"
            << "TradeType" << std::endl;
    outFile << std::string(100, '-') << std::endl;

    for (const auto& order : orders) {
        outFile << std::left << std::setw(40) << order.orderId
                << std::setw(15) << std::fixed << std::setprecision(6) << order.price
                << std::setw(10) << order.quantity
                << std::setw(25) << formatTimestamp(order.timestamp)
                << (order.type == TradeType::Buy ? "Buy" : "Sell") << std::endl;
    }
    outFile.close();
    std::cout << "Trade orders written to " << filename << std::endl;
}

void outputSummary(double elapsedTime, double speedup, double efficiency, const std::string& filename) {
    std::ofstream summaryFile(filename);
    if (!summaryFile.is_open()) {
        std::cerr << "Error opening summary file." << std::endl;
        return;
    }

    summaryFile << "Sorting Summary" << std::endl;
    summaryFile << "----------------" << std::endl;
    summaryFile << "Total execution time: " << elapsedTime << " seconds" << std::endl;
    summaryFile << "Speedup: " << speedup << std::endl;
    summaryFile << "Efficiency: " << efficiency << std::endl;

    summaryFile.close();
    std::cout << "Summary written to " << filename << std::endl;
}

// radix sorting
void radixSortByQuantity(std::vector<TradeOrder>& orders) {
    const int BASE = 10;
    std::vector<TradeOrder> buffer(orders.size());
    int maxQuantity = 0;

    for (const auto& order : orders) {
        maxQuantity = std::max(maxQuantity, order.quantity);
    }

    for (int exp = 1; maxQuantity / exp > 0; exp *= BASE) {
        std::vector<int> count(BASE, 0);
        for (const auto& order : orders) {
            count[(order.quantity / exp) % BASE]++;
        }
        for (int i = 1; i < BASE; ++i) {
            count[i] += count[i - 1];
        }
        for (int i = orders.size() - 1; i >= 0; --i) {
            int idx = (orders[i].quantity / exp) % BASE;
            buffer[--count[idx]] = orders[i];
        }

        // use move semantics to avoid unnecessary copies during reassignment
        orders = std::move(buffer);
        buffer.resize(orders.size());
    }
}

//merge sort
void mergeSortByPrice(std::vector<TradeOrder>& orders, int start, int end) {
    if (start >= end) return;
    int mid = start + (end - start) / 2;
    mergeSortByPrice(orders, start, mid);
    mergeSortByPrice(orders, mid + 1, end);
    std::inplace_merge(orders.begin() + start, orders.begin() + mid + 1, orders.begin() + end + 1,
                       [](const TradeOrder& a, const TradeOrder& b) {
                           return a.price < b.price;
                       });
}

//hybrid sort
void hybridSort(std::vector<TradeOrder>& orders) {
    radixSortByQuantity(orders);
    int start = 0;
    while (start < orders.size()) {
        int end = start;
        while (end < orders.size() && orders[end].quantity == orders[start].quantity)
            ++end;
        mergeSortByPrice(orders, start, end - 1);
        start = end;
    }
}

void dynamicLoadBalancing(std::vector<TradeOrder>& orders, int rank, int size, MPI_Comm comm) {
    int chunkSize = std::max((int)orders.size() / (4 * size), 5000); // Larger chunks for fewer communications
    std::vector<TradeOrder> localOrders;

    if (rank == 0) {
        // Master distributes chunks
        int orderIndex = 0;
        for (int i = 1; i < size && orderIndex < orders.size(); ++i) {
            int startIdx = orderIndex;
            int endIdx = std::min(orderIndex + chunkSize, (int)orders.size());
            orderIndex = endIdx;

            // Serialize and send chunk
            std::vector<char> chunkBuffer;
            for (int j = startIdx; j < endIdx; ++j) {
                std::vector<char> serializedOrder = orders[j].serialize();
                chunkBuffer.insert(chunkBuffer.end(), serializedOrder.begin(), serializedOrder.end());
            }
            int chunkSizeBytes = chunkBuffer.size();
            MPI_Send(&chunkSizeBytes, 1, MPI_INT, i, 0, comm); // Send chunk size
            MPI_Send(chunkBuffer.data(), chunkSizeBytes, MPI_BYTE, i, 1, comm); // Send chunk
        }

        // Process remaining orders locally
        int remainingStart = orderIndex;
        int remainingEnd = orders.size();
        for (int i = remainingStart; i < remainingEnd; ++i) {
            localOrders.push_back(orders[i]);
        }
        hybridSort(localOrders);

        // Receive sorted chunks from workers
        for (int i = 1; i < size; ++i) {
            MPI_Status status;
            int chunkSizeBytes;
            MPI_Recv(&chunkSizeBytes, 1, MPI_INT, i, 2, comm, &status); // Receive size
            std::vector<char> buffer(chunkSizeBytes);
            MPI_Recv(buffer.data(), chunkSizeBytes, MPI_BYTE, i, 3, comm, MPI_STATUS_IGNORE); // Receive chunk

            size_t offset = 0;
            while (offset < buffer.size()) {
                TradeOrder order;
                std::vector<char> orderBuffer(buffer.begin() + offset, buffer.end());
                order.deserialize(orderBuffer);
                localOrders.push_back(order);
                offset += order.serialize().size(); // Update offset
            }
        }

        // Final sort
        hybridSort(localOrders);
        orders = std::move(localOrders);

        // Send termination signals
        int terminationToken = -1;
        for (int i = 1; i < size; ++i) {
            MPI_Send(&terminationToken, 1, MPI_INT, i, 99, comm); // Tag 99: Termination
        }

    } else {
        // Worker processes
        while (true) {
            MPI_Status status;
            int chunkSizeBytes;
            MPI_Recv(&chunkSizeBytes, 1, MPI_INT, 0, MPI_ANY_TAG, comm, &status);

            if (status.MPI_TAG == 99) break; // Termination signal

            std::vector<char> buffer(chunkSizeBytes);
            MPI_Recv(buffer.data(), chunkSizeBytes, MPI_BYTE, 0, 1, comm, MPI_STATUS_IGNORE);

            // Deserialize orders
            std::vector<TradeOrder> chunkOrders;
            size_t offset = 0;
            while (offset < buffer.size()) {
                TradeOrder order;
                std::vector<char> orderBuffer(buffer.begin() + offset, buffer.end());
                order.deserialize(orderBuffer);
                chunkOrders.push_back(order);
                offset += order.serialize().size();
            }

            // Sort locally
            hybridSort(chunkOrders);

            // Serialize and send back sorted orders
            std::vector<char> sortedBuffer;
            for (const auto& order : chunkOrders) {
                std::vector<char> serializedOrder = order.serialize();
                sortedBuffer.insert(sortedBuffer.end(), serializedOrder.begin(), serializedOrder.end());
            }
            int sortedSizeBytes = sortedBuffer.size();
            MPI_Send(&sortedSizeBytes, 1, MPI_INT, 0, 2, comm); // Tag 2: Sorted chunk size
            MPI_Send(sortedBuffer.data(), sortedSizeBytes, MPI_BYTE, 0, 3, comm); // Tag 3: Sorted chunk
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numberOfOrders = 5'000'000;
    std::vector<TradeOrder> orders;

    double serialTime = 0.0;
    if (rank == 0) {
        std::cout << "Generating trade orders...\n";
        orders = generateTradeOrders(numberOfOrders);

        auto serialStart = MPI_Wtime();
        std::sort(orders.begin(), orders.end());
        auto serialEnd = MPI_Wtime();
        serialTime = serialEnd - serialStart;
        std::cout << "Serial execution time (T1): " << serialTime << " seconds\n";
    }

    MPI_Bcast(&serialTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        orders.resize(numberOfOrders);
    }

    MPI_Bcast(&numberOfOrders, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        orders.resize(numberOfOrders);
    }

    auto start = MPI_Wtime();
    dynamicLoadBalancing(orders, rank, size, MPI_COMM_WORLD);
    auto end = MPI_Wtime();

    if (rank == 0) {
        double elapsedTime = end - start;
        double speedup = serialTime / elapsedTime;
        double efficiency = speedup / size;

        outputTradeOrders(orders, "sorted_trade_orders.txt");
        outputSummary(elapsedTime, speedup, efficiency, "sorting_summary.txt");
    }

    MPI_Finalize();
    return 0;
}
